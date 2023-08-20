# Based on code from https://github.com/VainF/Torch-Pruning and https://github.com/pytorch/examples/tree/main/imagenet

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torch_pruning as tp
import engine.utils as utils
import registry
import copy


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, choices=["pretrain", "prune", "test"], default='prune')
parser.add_argument("--model", type=str, default='resnet56')
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="cifar10", choices=['cifar10', 'cifar100', 'modelnet40'])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--total-epochs", type=int, default=2)
parser.add_argument("--lr-decay-milestones", default="40,60", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default='cifar10_resnet56.pth')
parser.add_argument('--output-dir', default='run', help='path where to save')

# For pruning
parser.add_argument("--method", type=str, default='group_norm')
parser.add_argument("--speed-up", type=float, default=2.55)
parser.add_argument("--max-sparsity", type=float, default=1.0)
parser.add_argument("--sparsity", type=float, default=0.25)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=5e-4)
parser.add_argument("--weight-decay", type=float, default=5e-4)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=True)
parser.add_argument("--sl-total-epochs", type=int, default=10, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
parser.add_argument("--sl-restore", type=str, default=None)
parser.add_argument("--iterative-steps", default=1, type=int)

args = parser.parse_args()

def get_module_by_name(model, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, model)

def get_in_channel_history(original_model, pruned_model, step_history):
    pruned_in_channels_history_dict = {}
    print("=> Start history generation")
    for i, history in enumerate(reversed(step_history)):
        for pruned_layer_name, b, out_channels_removed in reversed(history):
            pruned_layer = get_module_by_name(pruned_model, pruned_layer_name)
            original_layer = get_module_by_name(original_model, pruned_layer_name)
            in_history = get_index_in_channel_history(original_layer, pruned_layer, out_channels_removed)
            pruned_in_channels_history_dict[pruned_layer_name] = in_history
            print(pruned_layer_name, in_history)

    return pruned_in_channels_history_dict

def get_index_in_channel_history(original_layer, pruned_layer, pruned_out_channels):
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
                                   pruned_layer.weight.data[out_channel_idx - skipped, in_channel_j, :, :]):
                        not_pruned_in_channels.append(in_channel_i)
                        continue
                        # in_channel_j of the pruned layer matches weights in the original layer, i.e not pruned

        all_channels = list(range(original_layer.in_channels))
        pruned_in_channels = [x for x in all_channels if x not in not_pruned_in_channels]
        # print(out_channel_idx, pruned_in_channels)
        # pruned_in_channels_history.append([out_channel_idx, pruned_in_channels])
        break  # the input channels dropped are the same for each output channel
    return pruned_in_channels

def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        print(current_speed_up)
    return current_speed_up

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()


# tune only the pruned weights
def train_freeze_model(
        model,
        train_loader,
        test_loader,
        epochs,
        lr,
        lr_decay_milestones,
        out_history,
        in_history,
        lr_decay_gamma=0.1,
        save_as=None,

        # For pruning
        weight_decay=5e-4,
        save_state_dict_only=True,
        pruner=None,
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if pruner is None else 0,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            temp_model = copy.deepcopy(model)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if pruner is not None:
                pruner.regularize(model)  # for sparsity learning
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
            # tune only weights
            for j, history in enumerate(reversed(out_history)):
                # loop through each layer changed in pruning

                for pruned_layer_name, b, out_channels_removed in reversed(history):
                        # loop through the layers of the larger model (same number of layers, different channel width)
                        for layer_name, tuning_params in model.named_parameters():

                            skipped_out_channels = 0

                            if layer_name == pruned_layer_name + ".weight":

                                temp_layer = get_module_by_name(temp_model, pruned_layer_name)
                                bigger_layer = get_module_by_name(model, pruned_layer_name)
                                in_channels_removed = in_history[pruned_layer_name]

                                # loop the channels of the bigger model
                                for out_channel_idx in range(bigger_layer.out_channels):
                                    # check if the channel has been dropped
                                    if out_channel_idx in out_channels_removed:
                                        # if channel was dropped, do not copy weights, use updated weights
                                        skipped_out_channels += 1
                                    else:
                                        # copy weights from tuned model to larger model
                                        if (bigger_layer.in_channels - temp_layer.in_channels) == 0:
                                            tuning_params.data[out_channel_idx, :, :, :] = \
                                                temp_layer.weight.data[out_channel_idx - skipped_out_channels, :, :, :]

                                        else:  # for conv layers with reshape of both input and output

                                            skipped_in_channels = 0

                                            for in_channel_idx in range(bigger_layer.in_channels):
                                                if in_channel_idx in in_channels_removed:
                                                    # if channel was dropped, do not copy weights from smaller tuned model
                                                    skipped_in_channels += 1
                                                else:
                                                    tuning_params.data[out_channel_idx, in_channel_idx, :, :] = \
                                                        temp_layer.weight.data[out_channel_idx - skipped_out_channels, in_channel_idx - skipped_in_channels, :, :]

        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join(args.output_dir,
                                           "{}_{}_{}.pth".format(args.dataset, args.model, args.method))

                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join(args.output_dir, "{}_{}.pth".format(args.dataset, args.model))
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))



def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    lr_decay_milestones,
    lr_decay_gamma=0.1,
    save_as=None,
    
    # For pruning
    weight_decay=5e-4,
    save_state_dict_only=True,
    pruner=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if pruner is None else 0,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if pruner is not None:
                pruner.regularize(model) # for sparsity learning
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
        
        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join(args.output_dir, "{}_{}_{}.pth".format(args.dataset, args.model, args.method) )

                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}.pth".format(args.dataset, args.model) )
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))

def get_layers(model: torch.nn.Module, parent_name=''):
    layers = {}
    for name, module in model.named_children():
        layer_name = f"{parent_name}.{name}" if parent_name else name
        if len(list(module.children())) == 0:
            layers[layer_name] = module
        else:
            layers.update(get_layers(module, parent_name=layer_name))
    return layers

def rebuild_model(tuned_model, bigger_model, verification_model, device, out_history, in_history):
    print("=> Starting rebuilding...")
    layers_total = 0
    layers_rebuilt_count = 0

    tuned_model = tuned_model.to(device)
    bigger_model = bigger_model.to(device)
    verification_layers = get_layers(verification_model)

    # for name, params in tuned_model.named_parameters():
    #     # if name == "conv1.weight":
    #     print(name)
    #     new_tensor = torch.ones_like(params.data)
    #     params.data = new_tensor

    # save_onnx(bigger_model, "before", device)
    for i, history in enumerate(reversed(out_history)):
        for pruned_layer_name, b, out_channels_removed in reversed(history):
            layers_total += 1

    for i, history in enumerate(reversed(out_history)):
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
                    verification_layer = get_module_by_name(verification_model, pruned_layer_name)
                    in_channels_removed = in_history[pruned_layer_name]

                    # loop throughout the channels of the bigger model
                    for out_channel_idx in range(bigger_layer.out_channels):

                        # check if the channel has been dropped
                        if out_channel_idx in out_channels_removed:
                            # if channel was dropped, do not copy weights from smaller tuned model
                            skipped_out_channels += 1

                        else:
                            # copy weights from tuned model to larger model
                            if (bigger_layer.in_channels - tuned_layer.in_channels) == 0:
                                bigger_layer_params.data[out_channel_idx, :, :, :] = \
                                    tuned_layer.weight.data[out_channel_idx - skipped_out_channels, :, :, :]


                            else:  # for conv layers with reshape of both input and output
                                skipped_in_channels = 0
                                for in_channel_idx in range(bigger_layer.in_channels):

                                    if in_channel_idx in in_channels_removed:
                                        # if channel was dropped, do not copy weights from smaller tuned model
                                        skipped_in_channels += 1
                                    else:
                                        bigger_layer_params.data[out_channel_idx, in_channel_idx, :, :] = \
                                            tuned_layer.weight.data[out_channel_idx - skipped_out_channels,
                                            in_channel_idx - skipped_in_channels, :, :]

                    layers_rebuilt_count += 1
                    print("(" + str(layers_rebuilt_count), "of", str(layers_total) + ")", pruned_layer_name, "has been rebuilt.")
                    # if torch.equal(bigger_layer_params.data, verification_layer.weight.data):
                    #     print("Same (Correct when pruned):", pruned_layer_name, bigger_layer)
                    # else:
                    #     print("Different (Correct when tuned):", pruned_layer_name, bigger_layer)

    return bigger_model

def get_pruner(model, example_inputs):
    args.sparsity_learning = False
    # if args.method == "random":
    #     imp = tp.importance.RandomImportance()
    #     pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    # elif args.method == "l1":
    #     imp = tp.importance.MagnitudeImportance(p=1)
    #     pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    # elif args.method == "lamp":
    #     imp = tp.importance.LAMPImportance(p=2)
    #     pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    # elif args.method == "slim":
    #     args.sparsity_learning = True
    #     imp = tp.importance.BNScaleImportance()
    #     pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    # elif args.method == "group_norm":
    #     imp = tp.importance.GroupNormImportance(p=2)
    #     pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    # elif args.method == "group_sl":
    #     args.sparsity_learning = True
    #     imp = tp.importance.GroupNormImportance(p=2)
    #     pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    # else:
    #     raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)

    # imp = tp.importance.GroupNormImportance(p=2)
    imp = tp.importance.MagnitudeImportance(p=1)
    #     pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    # pruner = tp.pruner.GroupNormPruner(
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        ch_sparsity=args.sparsity,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        global_pruning=args.global_pruning
    )
    return pruner


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Logger
    if args.mode == "prune":
        prefix = 'global' if args.global_pruning else 'local'
        logger_name = "{}-{}-{}-{}".format(args.dataset, prefix, args.method, args.model)
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode, logger_name)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "pretrain":
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode)
        logger_name = "{}-{}".format(args.dataset, args.model)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "test":
        log_file = None
    args.logger = utils.get_logger(logger_name, output=log_file)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, train_dst, val_dst, input_size = registry.get_dataset(
        args.dataset, data_root="../../data"
    )
    args.num_classes = num_classes
    pretrained_model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset)


    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=args.batch_size, num_workers=4
    )
    
    for k, v in utils.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            pretrained_model = loaded
        else:
            pretrained_model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))

    pruned_model = copy.deepcopy(pretrained_model)  # this model will be pruned
    rebuilt_model = copy.deepcopy(pretrained_model)

    pretrained_model = pretrained_model.to(args.device)
    pruned_model = pruned_model.to(args.device)
    rebuilt_model = rebuilt_model.to(args.device)


    ######################################################
    # Training / Pruning / Testing
    example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    if args.mode == "pretrain":
        ops, params = tp.utils.count_ops_and_params(
            pruned_model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        train_model(
            model=pruned_model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader
        )
    elif args.mode == "prune":
        pruner = get_pruner(pruned_model, example_inputs=example_inputs)
        # 0. Sparsity Learning
        if args.sparsity_learning:
            reg_pth = "reg_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.reg)
            reg_pth = os.path.join( os.path.join(args.output_dir, reg_pth) )
            if not args.sl_restore:
                args.logger.info("Regularizing...")
                train_model(
                    pruned_model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.sl_total_epochs,
                    lr=args.sl_lr,
                    lr_decay_milestones=args.sl_lr_decay_milestones,
                    lr_decay_gamma=args.lr_decay_gamma,
                    pruner=pruner,
                    save_state_dict_only=True,
                    save_as = reg_pth,
                )
            args.logger.info("Loading the sparse model from {}...".format(reg_pth))
            pruned_model.load_state_dict( torch.load( reg_pth, map_location=args.device) )

        # 1. Pruning
        pruned_model.eval()

        # print(pruned_model.conv1.weight.data[0])
        # print(pruned_model.conv1.weight.data[1])

        # args.logger.info("Validating...")
        # acc, val_loss = eval(model, test_loader)
        # args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))

        ori_ops, ori_size = tp.utils.count_ops_and_params(pruned_model, example_inputs=example_inputs)
        ori_acc, ori_val_loss = eval(pruned_model, test_loader, device=args.device)
        args.logger.info("Pruning...")
        pruner.step()

        history = pruner.pruning_history()
        layers_affected = len(history)
        layers_affected_per_step = int(layers_affected / args.iterative_steps)
        out_history = [history[i:i + layers_affected_per_step] for i in
                       range(0, layers_affected, layers_affected_per_step)]

        in_history = {}
        in_history = get_in_channel_history(pretrained_model, pruned_model, out_history)
        args.logger.info(pruned_model)
        args.logger.info("Validating partial prune...")
        acc_partial_prune, val_loss_partial_prune = eval(pruned_model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc_partial_prune, val_loss_partial_prune))
        pruned_ops_01, pruned_size = tp.utils.count_ops_and_params(pruned_model, example_inputs=example_inputs)


        args.logger.info("Pruning 02...")
        partial_rebuilt_model = copy.deepcopy(pruned_model)
        pruned_model_02 = copy.deepcopy(pruned_model)
        pruned_model_02 = pruned_model_02.to(args.device)
        pruned_model_02.eval()
        pruner_02 = get_pruner(pruned_model_02, example_inputs=example_inputs)
        pruner_02.step()


        history_02 = pruner_02.pruning_history()
        layers_affected_02 = len(history_02)
        layers_affected_per_step_02 = int(layers_affected_02 / args.iterative_steps)
        out_history_02 = [history_02[i:i + layers_affected_per_step_02] for i in
                       range(0, layers_affected_02, layers_affected_per_step_02)]

        in_history_02 = {}
        in_history_02 = get_in_channel_history(pruned_model, pruned_model_02, out_history_02)

        # del pruner # remove reference
        args.logger.info(pruned_model_02)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(pruned_model_02, example_inputs=example_inputs)
        pruned_acc, pruned_val_loss = eval(pruned_model_02, test_loader, device=args.device)
        
        args.logger.info(
            "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
            )
        )
        args.logger.info(
            "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops / 1e6,
                pruned_ops_01 / 1e6,
                pruned_ops_01 / ori_ops * 100,
                ori_ops / pruned_ops_01,
            )
        )
        args.logger.info(
            "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops / 1e6,
                pruned_ops / 1e6,
                pruned_ops / ori_ops * 100,
                ori_ops / pruned_ops,
            )
        )
        args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
        args.logger.info(
            "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
        )

        # model_statistics_base = summary(pretrained_model, example_inputs, depth=3,
        #                            col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], )
        # model_statistics_str_base = str(model_statistics_base)
        
        tuned_model = copy.deepcopy(pruned_model_02)
        # 2. Finetuning
        args.logger.info("Finetuning...")
        train_model(
            tuned_model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            save_state_dict_only=False,
        )

        pruned_model_02.eval()
        ops, params = tp.utils.count_ops_and_params(
            tuned_model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        acc, val_loss = eval(tuned_model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))

        args.logger.info("Partial Rebuilding...")
        # tuned, bigger, verif

        partial_rebuilt_model = rebuild_model(tuned_model, partial_rebuilt_model, partial_rebuilt_model, args.device, out_history_02, in_history_02)

        args.logger.info("Validating partial rebuilt...")
        acc_partial_rebuilt, val_loss_partial_rebuilt = eval(partial_rebuilt_model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc_partial_rebuilt, val_loss_partial_rebuilt))

        tuned_partial_rebuilt_model = copy.deepcopy(partial_rebuilt_model)
        args.logger.info("Finetuning partial rebuilt...")
        train_freeze_model(
            tuned_partial_rebuilt_model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            save_state_dict_only=False,
            out_history=out_history_02,
            in_history=in_history_02
        )

        args.logger.info("Final Rebuilding...")
        # tuned, bigger, verif
        rebuilt_model = rebuild_model(tuned_partial_rebuilt_model, rebuilt_model, pretrained_model, args.device, out_history, in_history)

        # print(rebuilt_model.conv1.weight.data[0])
        # print(rebuilt_model.conv1.weight.data[1])
        args.logger.info("Validating final rebuilt...")
        acc_rebuilt, val_loss_rebuilt = eval(rebuilt_model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc_rebuilt, val_loss_rebuilt))

        final_model = copy.deepcopy(rebuilt_model)

        # 2. Finetuning
        args.logger.info("Fine tuning rebuilt...")
        # train_freeze_model(
        #     final_model,
        #     epochs=args.total_epochs,
        #     lr=args.lr,
        #     lr_decay_milestones=args.lr_decay_milestones,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     device=args.device,
        #     save_state_dict_only=False,
        #     out_history=out_history,
        #     in_history=in_history
        # )
        train_model(
            final_model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            save_state_dict_only=False,
        )

if __name__ == "__main__":
    main()
