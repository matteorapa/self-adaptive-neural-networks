import torch
from enum import Enum
from functools import reduce
from torch.utils.data import Subset
import time
import copy
import torch.distributed as dist
import argparse
import os
import random
import warnings
import pickle
from torchvision.models import ResNet50_Weights, ResNet18_Weights, resnet18, resnet50
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import torch_pruning as tp
from torchinfo import summary

# Code bases on pytorch's imagenet example: https://github.com/pytorch/examples/tree/main/imagenet


def save_onnx(model, filename, device):
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model,
                      example_inputs,
                      '../checkpoints/onnx/'+filename+".onnx",
                      verbose=False,
                      input_names=["Input Image"],
                      output_names=["Classification Probabilities"],
                      export_params=True,
                      )


def load_model(path, pruned=False):
    if pruned:
        model = resnet50()
        state = torch.load('../checkpoints/'+path+".pth", map_location='cpu')
        tp.load_state_dict(model, state_dict=state)
        return model
    else:
        model = resnet50()
        model.load_state_dict(torch.load('../checkpoints/'+path+".pth"))


def save_model(model, filename, pruned=False):
    if pruned:
        state_dict = tp.state_dict(model)  # the pruned model
        torch.save(state_dict, '../checkpoints/'+filename+".pth")
    else:
        torch.save(model.state_dict(), '../checkpoints/'+filename+".pth")


def load_out_history(path):
    with open('../checkpoints/history/out_'+path, 'rb') as file:
        history_steps = pickle.load(file)
    return history_steps


def load_in_history(filename):
    with open('../checkpoints/history/in_'+filename+'.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def save_out_history(history_steps, filename):
    with open('../checkpoints/history/out_'+filename, 'wb') as temp:
        pickle.dump(history_steps, temp)


def save_in_history(d, filename):
    with open('../checkpoints/history/in_'+filename+'.pkl', 'wb') as f:
        pickle.dump(d, f)


def get_module_by_name(model, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, model)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

def validate(val_loader, model, criterion, args, device):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg

# Function generated using chatGPT
def get_layers(model: torch.nn.Module, parent_name=''):
    layers = {}
    for name, module in model.named_children():
        layer_name = f"{parent_name}.{name}" if parent_name else name
        if len(list(module.children())) == 0:
            layers[layer_name] = module
        else:
            layers.update(get_layers(module, parent_name=layer_name))
    return layers

def show_layers(model: torch.nn.Module):
    layers = get_layers(model)
    for i, (name, module) in enumerate(layers.items()):
        if hasattr(module, 'weight'):
            print( name, module)

def compare_models(model_a, model_b):
    layers_a = get_layers(model=model_a)
    layers_b = get_layers(model=model_b)
    i = 1
    layer_count = len(layers_a)
    if (len(layers_a) == len(layers_b)):
        print("Same layer count")

    for i, (name, module) in enumerate(layers_a.items()):
        if hasattr(module, 'weight'):
            if not torch.equal(module.weight.data, layers_b[name].weight.data):
                print(str(i), "of", str(layer_count), "Fail:", name, module)
        i += 1
def get_in_channel_history(original_model, pruned_model, step_history):
    pruned_in_channels_history_dict = {}
    print("=> Start history generation")
    for i, history in enumerate(reversed(step_history)):
        for pruned_layer_name, b, out_channels_removed in reversed(history):
            print(pruned_layer_name)
            pruned_layer = get_module_by_name(pruned_model, pruned_layer_name)
            original_layer = get_module_by_name(original_model, pruned_layer_name)
            in_history = get_index_in_channel_history(original_layer, pruned_layer, out_channels_removed)
            pruned_in_channels_history_dict[pruned_layer_name] = in_history
            print(in_history)

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
        print(out_channel_idx, pruned_in_channels)
        # pruned_in_channels_history.append([out_channel_idx, pruned_in_channels])
        break  # the input channels dropped are the same for each output channel
    return pruned_in_channels

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
