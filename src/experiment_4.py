import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import copy
from functools import reduce
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
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
from torch.utils.data import Subset
import torch_pruning as tp
from torchinfo import summary

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='../data',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--prune', default=None, type=float,
                    help='The amount to prune.')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # validate(val_loader, model, criterion, args)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    def get_module_by_name(model, access_string):
        names = access_string.split(sep='.')
        return reduce(getattr, names, model)

    def save_checkpoint(state, filepath, name):
        torch.save(state, os.path.join(filepath, name+'checkpoint.pth'))

    # print("=> evaluate model before pruning")
    # validate(val_loader, model, criterion, args)

    sparsity = 0.25
    print("Pruning sparsity:", sparsity)
    model.eval()

    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.TaylorImportance()

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    iterative_steps = 1

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        round_to=None,
        unwrapped_parameters=None,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity = sparsity,
        ignored_layers=ignored_layers,
    )


    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    pruned_models = []
    pruned_models.append(copy.deepcopy(model))
    pruned_models[-1].to(device)

    for i in range(iterative_steps):
        print("=> pruning model iteration", i)
        if isinstance(imp, tp.importance.TaylorImportance):
            loss = model(example_inputs).sum()
            loss.backward()
        pruner.step()
        pruned_models.append(copy.deepcopy(model))
        pruned_models[-1].to(device)

        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
    print("=> evaluate pruned model (No fine-tuning)")
    validate(val_loader, pruned_models[0], criterion, args)

    print("=> starting fine-tuning of pruned model...")
    for epoch in range(0, 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(val_loader, pruned_models[1], criterion, optimizer, epoch, device, args, step_history=None)
        scheduler.step()

    print("=> evaluate pruned and tuned model")
    # validate model after pruning and tuning
    validate(val_loader, pruned_models[0], criterion, args)
    torch.save(tp.state_dict(pruned_models[0]),"exp_3_model_resnet18_pruned_0.3_tuned.pth")

    # model = torch.load('exp_3_model_resnet18_pruned_0.3_tuned.pth')

    # iterative_steps
    history = pruner.pruning_history()
    layers_affected = len(history)
    layers_affected_per_step = int(layers_affected / iterative_steps)
    step_history = [history[i:i+layers_affected_per_step] for i in range(0, layers_affected, layers_affected_per_step)]
    pruned_models.reverse() # 60, 61, 62, 63, 64
    with open('0.3_history_exp3', 'wb') as data:
        pickle.dump(step_history, data)

    print("=> rebuild pruned and tuned model to original size")

    for i, history in enumerate(reversed(step_history)):
        tuned_model = pruned_models[i]
        bigger_model = pruned_models[i+1]

        # loop through each layer changed in pruning
        for pruned_layer_name, b, channels_removed in reversed(history):
            # loop through the layers of the larger model (same number of layers, different channel width)
            for layer_name, bigger_layer_params in bigger_model.named_parameters():

                skipped = 0

                if(layer_name == pruned_layer_name+".weight"):

                        tuned_layer = get_module_by_name(tuned_model, pruned_layer_name)
                        bigger_layer = get_module_by_name(bigger_model, pruned_layer_name)


                        # loop throught the channels of the bigger model
                        for idx in range(bigger_layer.out_channels):

                            # check if the channel has been dropped
                            if idx in channels_removed:
                                # if channel was dropped, do not copy weights from smaller tuned model
                                # print("Channel was skipped:", channels_removed[skipped])
                                skipped += 1

                            else:
                                # copy weights from tuned model to larger model
                                if "layer" not in layer_name:

                                    bigger_layer_params.requires_grad_(False)
                                    bigger_layer_params[idx,:, : ,:] = tuned_layer.weight.data[idx-skipped,:, : ,:]

                                else: # for conv layers with reshape of both input and output

                                    bigger_layer_params.requires_grad_(False)
                                    skipped_j = 0

                                    if (bigger_layer.in_channels - tuned_layer.in_channels) == len(channels_removed):

                                        for idx_j in range(bigger_layer.in_channels):

                                            if idx_j in channels_removed:
                                                # if channel was dropped, do not copy weights from smaller tuned model
                                                skipped_j += 1
                                            else:
                                                bigger_layer_params[idx,idx_j, : ,:] = tuned_layer.weight.data[idx-skipped,idx_j-skipped_j, : ,:]

    # validate model after rebuilding, degraded accuracy expected
    print("=> evaluate rebuilt model (no fine-tuning)")
    validate(val_loader, pruned_models[1], criterion, args)
    torch.save(tp.state_dict(pruned_models[0]), "exp_3_model_resnet18_rebuilt_0.3.pth")

    # create a new model, e.g. resnet18
    # model = models.__dict__[args.arch](pretrained=True)
    #
    # # load the pruned state_dict into the unpruned model.
    # loaded_state_dict = torch.load('exp_3_model_resnet18_rebuilt_0.3.pth', map_location='cpu')
    # tp.load_state_dict(model, state_dict=loaded_state_dict)
    # # model = torch.load('exp_3_model_resnet18_rebuilt_0.3.pth')
    # model.to(device)

    # with open('0.3_history_exp3', 'rb') as file:
    #     step_history = pickle.load(file)

    print("=> Starting fine-tuning of rebuilt model (Only train tune pruned channels)...")
    for epoch in range(0, 10):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(val_loader, model, criterion, optimizer, epoch, device, args, step_history)
        scheduler.step()
        print("=> evaluate rebuilt and tuned model")
        validate(val_loader, model, criterion, args)

    torch.save(tp.state_dict(model), "exp_3_model_resnet18_rebuilt_0.3_tuned.pth")

    model_statistics = summary(model, (1, 3, 224, 224), depth=3,
                               col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"])

def get_module_by_name(model, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, model)

def train(train_loader, model, criterion, optimizer, epoch, device, args, step_history):
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

        temp_model = copy.deepcopy(model)
        optimizer.step()

        # for i, param in enumerate(optimizer.param_groups[0]['params']):
        #     print(i, param.grad.shape)
        #     param.grad = torch.zeros_like(param.grad)
        #     gradients update don't work due to momentum
        # loop through each layer changed in pruning

        if step_history is not None:
            for history in reversed(step_history):
                for pruned_layer_name, b, channels_removed in reversed(history):
                    # loop through the layers of the larger model (same number of layers, different channel width)
                    for layer_name, tuning_params in model.named_parameters():

                        skipped = 0

                        if (layer_name == pruned_layer_name + ".weight"):

                            temp_layer = get_module_by_name(temp_model, pruned_layer_name)
                            bigger_layer = get_module_by_name(model, pruned_layer_name)

                            # loop throught the channels of the bigger model
                            for idx in range(bigger_layer.out_channels):

                                # check if the channel has been dropped
                                if idx in channels_removed:
                                    # if channel was dropped, do not copy weights, use updated weights

                                    skipped += 1

                                else:

                                    # the non pruned channels are not updated, and copied back from the temp weights before weight updates
                                    if "layer" not in layer_name:

                                        tuning_params.requires_grad_(False)
                                        # the non pruned channels are not updated, and copied back from the temp weights
                                        #                         # before weight updates
                                        tuning_params[idx, :, :, :] = temp_layer.weight.data[idx - skipped, :, :, :]

                                    else:  # for conv layers with reshape of both input and output

                                        tuning_params.requires_grad_(False)
                                        skipped_j = 0

                                        if (bigger_layer.in_channels - temp_layer.in_channels) == len(channels_removed):

                                            for idx_j in range(bigger_layer.in_channels):

                                                if idx_j in channels_removed:
                                                    # if channel was dropped, do not copy weights from smaller tuned model
                                                    skipped_j += 1
                                                else:
                                                    tuning_params[idx, idx_j, :, :] = temp_layer.weight.data[idx - skipped, idx_j - skipped_j, :, :]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

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


if __name__ == '__main__':
    main()
