from utils import *

# todo save history beforehand
def get_layer_in_channel_history(original, pruned, layer, pruned_out_channels):
    original_layer = get_module_by_name(original, layer)
    pruned_layer = get_module_by_name(pruned, layer)
    skipped = 0  # adjustment to match out_channel between original and pruned model of different shapes
    pruned_in_channels_history = []

    for out_channel_idx in range(original_layer.out_channels):
        not_pruned_in_channels = []  # in channels pruned per out channel
        if out_channel_idx in pruned_out_channels:
            # the out_channel is completely pruned
            skipped += 1
        else:
            for in_channel_i in range(original_layer.in_channels):
                # the out_channel is partially pruned, loop through the in channels
                # and find which idx have been pruned for each non-pruned out channel
                for in_channel_j in range(pruned_layer.in_channels):
                    # the output channel exists in both pruned and original model
                    if torch.equal(original_layer.weight.data[out_channel_idx, in_channel_i, :, :],
                                   original_layer.weight.data[out_channel_idx - skipped, in_channel_j, :, :]):
                        not_pruned_in_channels.append(in_channel_j)
                        continue
                        # in_channel_j of the pruned layer matches weights in the original layer, i.e not pruned

        all_channels = list(range(original_layer.in_channels))
        pruned_in_channels = [x for x in all_channels if x not in not_pruned_in_channels]
        pruned_in_channels_history.append([out_channel_idx, pruned_in_channels])
        break  # the input channels dropped are the same for each output channel
    return pruned_in_channels_history

# todo update
def rebuild_model(tuned_model, bigger_model, step_history):
    for i, history in enumerate(reversed(step_history[0])):

        # loop through each layer changed in pruning
        for pruned_layer_name, b, out_channels_removed in reversed(history):

            # loop through the layers of the larger model (same number of layers, different channel width)
            for layer_name, bigger_layer_params in bigger_model.named_parameters():

                skipped = 0
                # and layer_name == "conv1.weight"
                if"module."+layer_name == pruned_layer_name+".weight":

                        # get copy of layers
                        tuned_layer = get_module_by_name(tuned_model, layer_name[:-7])
                        bigger_layer = get_module_by_name(bigger_model, layer_name[:-7])

                        # if (bigger_layer.in_channels - tuned_layer.in_channels) != len(out_channels_removed):
                        #     print(bigger_layer.out_channels, bigger_layer.in_channels, layer_name, tuned_layer.out_channels, tuned_layer.in_channels, len(out_channels_removed))
                        #
                        # # print(layer_name, out_channels_removed)

                        # loop throught the channels of the bigger model
                        for idx in range(bigger_layer.out_channels):

                            # check if the channel has been dropped
                            if idx in out_channels_removed:
                                # if channel was dropped, do not copy weights from smaller tuned model
                                # print("Channel was skipped")
                                skipped += 1

                            else:
                                get_layer_in_channel_history(bigger_layer, tuned_layer, layer_name, out_channels_removed)
                                # copy weights from tuned model to larger model
                                if "layer" not in layer_name:
                                    # print(layer_name, idx)
                                    bigger_layer_params.data[idx, :, :,:] = tuned_layer.weight.data[idx-skipped, :, :, :]

                                else: # for conv layers with reshape of both input and output

                                    # bigger_layer_params.requires_grad_(False)
                                    skipped_j = 0

                                    if (bigger_layer.in_channels - tuned_layer.in_channels) == len(out_channels_removed):

                                        for idx_j in range(bigger_layer.in_channels):

                                            if idx_j in out_channels_removed:
                                                # if channel was dropped, do not copy weights from smaller tuned model
                                                skipped_j += 1
                                            else:
                                                bigger_layer_params.data[idx,idx_j, : ,:] = tuned_layer.weight.data[idx-skipped,idx_j-skipped_j, : ,:]

    return bigger_model


# def rebuild_model(tuned_model, bigger_model, step_history):
#     for i, history in enumerate(reversed(step_history[0])):
#
#         # loop through each layer changed in pruning
#         for pruned_layer_name, b, channels_removed in reversed(history):
#
#             # loop through the layers of the larger model (same number of layers, different channel width)
#             for layer_name, bigger_layer_params in bigger_model.named_parameters():
#
#                 skipped = 0
#                 # and layer_name == "conv1.weight"
#                 if"module."+layer_name == pruned_layer_name+".weight":
#
#                         # get copy of layers
#                         tuned_layer = get_module_by_name(tuned_model, layer_name[:-7])
#                         bigger_layer = get_module_by_name(bigger_model, layer_name[:-7])
#
#                         if (bigger_layer.in_channels - tuned_layer.in_channels) != len(channels_removed):
#                             print(bigger_layer.out_channels, bigger_layer.in_channels, layer_name, tuned_layer.out_channels, tuned_layer.in_channels, len(channels_removed))
#
#                         # print(layer_name, channels_removed)
#
#                         # loop throught the channels of the bigger model
#                         for idx in range(bigger_layer.out_channels):
#
#                             # check if the channel has been dropped
#                             if idx in channels_removed:
#                                 # if channel was dropped, do not copy weights from smaller tuned model
#                                 # print("Channel was skipped")
#                                 skipped += 1
#
#                             else:
#                                 # copy weights from tuned model to larger model
#                                 if "layer" not in layer_name:
#                                     # print(layer_name, idx)
#                                     bigger_layer_params.data[idx,:, : ,:] = tuned_layer.weight.data[idx-skipped,:, : ,:]
#
#                                 else: # for conv layers with reshape of both input and output
#
#                                     # bigger_layer_params.requires_grad_(False)
#                                     skipped_j = 0
#
#                                     if (bigger_layer.in_channels - tuned_layer.in_channels) == len(channels_removed):
#
#                                         for idx_j in range(bigger_layer.in_channels):
#
#                                             if idx_j in channels_removed:
#                                                 # if channel was dropped, do not copy weights from smaller tuned model
#                                                 skipped_j += 1
#                                             else:
#                                                 bigger_layer_params.data[idx,idx_j, : ,:] = tuned_layer.weight.data[idx-skipped,idx_j-skipped_j, : ,:]
#
#     return bigger_model