from utils import *

def get_layer_in_channel_history(original_layer, pruned_layer, layer, pruned_out_channels):
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
        # pruned_in_channels_history.append([out_channel_idx, pruned_in_channels])
        break  # the input channels dropped are the same for each output channel
    return pruned_in_channels

def get_index_in_channel_history(original_layer, pruned_layer, pruned_out_channels):
    skipped = 0  # adjustment to match out_channel between original and pruned model of different shapes
    pruned_in_channels_history = []
    print(pruned_layer)

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
                                   pruned_layer.weight.data[out_channel_idx - skipped, in_channel_j, :, :]):
                        not_pruned_in_channels.append(in_channel_i)
                        continue
                        # in_channel_j of the pruned layer matches weights in the original layer, i.e not pruned

        all_channels = list(range(original_layer.in_channels))
        pruned_in_channels = [x for x in all_channels if x not in not_pruned_in_channels]
        print(out_channel_idx, pruned_in_channels)
        # pruned_in_channels_history.append([out_channel_idx, pruned_in_channels])
        break  # the input channels dropped are the same for each output channel
    return pruned_in_channels

def get_in_channel_history(original_model, pruned_model, step_history):
    pruned_in_channels_history_dict = {}
    for i, history in enumerate(reversed(step_history)):
        for pruned_layer_name, b, out_channels_removed in reversed(history):
            pruned_layer = get_module_by_name(pruned_model, pruned_layer_name)
            original_layer = get_module_by_name(original_model, pruned_layer_name)
            skipped = 0  # adjustment to match out_channel between original and pruned model of different shapes


            for out_channel_idx in range(original_layer.out_channels):
                not_pruned_in_channels = []  # in channels pruned per out channel
                if out_channel_idx in out_channels_removed:
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
                pruned_in_channels_history_dict[pruned_layer_name] = pruned_in_channels
                print(pruned_layer_name, pruned_in_channels)
                # pruned_in_channels_history.append([out_channel_idx, pruned_in_channels])
                break  # the input channels dropped are the same for each output channel
    return pruned_in_channels_history_dict

def rebuild_model(tuned_model, bigger_model, device):
    print("=> Starting rebuilding...")
    layers_total = 0
    layers_rebuilt_count = 0

    out_step_history = load_out_history("resnet50_pruned_0.125")
    in_channels_history = load_in_history("resnet50_pruned_0.125")

    tuned_model = tuned_model.to(device)
    bigger_model = bigger_model.to(device)
    verification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    verification_layers = get_layers(verification_model)

    for i, history in enumerate(reversed(out_step_history)):
        for pruned_layer_name, b, out_channels_removed in reversed(history):
            layers_total += 1

    for i, history in enumerate(reversed(out_step_history)):
        # loop through each layer changed in pruning

        for pruned_layer_name, b, out_channels_removed in reversed(history):

            # loop through the layers of the larger model (same number of layers, different channel width)
            for layer_name, bigger_layer_params in bigger_model.named_parameters():

                skipped_out_channels = 0
                # if"module."+layer_name == pruned_layer_name+".weight":
                # if layer_name == pruned_layer_name + ".weight" and pruned_layer_name == "layer1.2.conv2":
                if layer_name == pruned_layer_name + ".weight":
                    # get copy of layers
                    tuned_layer = get_module_by_name(tuned_model, pruned_layer_name)
                    bigger_layer = get_module_by_name(bigger_model, pruned_layer_name)
                    verification_layer = get_module_by_name(bigger_model, pruned_layer_name)
                    # in_channels_removed = get_layer_in_channel_history(bigger_layer, tuned_layer, layer_name, out_channels_removed)
                    in_channels_removed_old = in_channels_history[pruned_layer_name]
                    in_channels_removed = get_index_in_channel_history(bigger_layer, tuned_layer, out_channels_removed)


                    # loop throughout the channels of the bigger model
                    for out_channel_idx in range(bigger_layer.out_channels):

                        # check if the channel has been dropped
                        if out_channel_idx in out_channels_removed:
                            # if channel was dropped, do not copy weights from smaller tuned model
                            skipped_out_channels += 1

                        else:
                            # copy weights from tuned model to larger model
                            if (bigger_layer.in_channels - tuned_layer.in_channels) == 0:
                                bigger_layer_params.data[out_channel_idx, :, :, :] =\
                                    tuned_layer.weight.data[out_channel_idx - skipped_out_channels, :, :, :]


                            else:  # for conv layers with reshape of both input and output
                                skipped_in_channels = 0
                                for in_channel_idx in range(bigger_layer.in_channels):

                                    if in_channel_idx in in_channels_removed:
                                        # if channel was dropped, do not copy weights from smaller tuned model
                                        skipped_in_channels += 1
                                    else:

                                        bigger_layer_params.data[out_channel_idx, in_channel_idx, :, :] = \
                                                tuned_layer.weight.data[out_channel_idx - skipped_out_channels, in_channel_idx - skipped_in_channels, :, :]

                                # if torch.equal(bigger_layer_params.data[out_channel_idx, in_channel_idx, :, :],
                                #        verification_layer.weight.data[out_channel_idx, in_channel_idx, :, :]):
                                #     print("Pass:", pruned_layer_name, out_channel_idx)
                                # else:
                                #     print("Fail:", pruned_layer_name, out_channel_idx)

                    if torch.equal(bigger_layer_params.data, verification_layers[pruned_layer_name].weight.data):
                        print("Pass:", pruned_layer_name, bigger_layer)
                    else:
                        print("Fail:", pruned_layer_name, bigger_layer)
            layers_rebuilt_count += 1
            print("("+str(layers_rebuilt_count), "of", str(layers_total)+")", pruned_layer_name, "has been rebuilt.")

    return bigger_model


def validate_layer(bigger_model, verif_model, layer_name):
    if torch.equal(bigger_model.state_dict()[layer_name], verif_model.state_dict()[layer_name]):
        print("Pass")
    else:
        print("Fail")

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