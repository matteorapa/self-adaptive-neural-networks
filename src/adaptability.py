import torch
from utils import *
from prune import *
from rebuild import *
from finetune_freeze import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='../data',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
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
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
    args.gpu = gpu

    print("=> using pre-trained model '{}'".format(args.arch))
    model_str = "resnet18"

    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # verification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # original_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #
    # if model_str == "resnet18":

    model = resnet18(weights=ResNet18_Weights)
    verification_model = resnet18(weights=ResNet18_Weights)
    original_model = resnet18(weights=ResNet18_Weights)

    device = torch.device("cuda:0")
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    model = model.to(device)
    verification_model = verification_model.to(device)
    original_model = original_model.to(device)

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(None is None),
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    criterion = nn.CrossEntropyLoss().to(device)

    # print("=> evaluate model")
    # validate(val_loader, model, criterion, args, device)
    prune = 0.05
    prune_02 = 0.05

    pruned_model, out_history, in_history = apply_channel_prune(model, original_model, prune, example_inputs)
    pruned_model_01 = deepcopy(pruned_model)
    pruned_model_01 = pruned_model_01.to(device)

    # save model 01, out_channels, in_channels
    save_model(pruned_model_01, model_str+"_pruned_" + str(prune), pruned=True)
    save_out_history(out_history, model_str+"_pruned_" + str(prune))
    save_in_history(in_history, model_str+"_pruned_" + str(prune))

    # print("=> evaluate pruned model 01")
    # validate(val_loader, pruned_model_01, criterion, args, device)

    pruned_model, out_history_02, in_history_02 = apply_channel_prune(pruned_model, pruned_model_01, prune_02, example_inputs)
    pruned_model_02 = deepcopy(pruned_model)
    pruned_model_02 = pruned_model_02.to(device)

    # save model 02, out_channels, in_channels
    save_model(pruned_model_02, model_str+"_pruned_" + str(prune) + "+" + str(prune_02), pruned=True)
    save_out_history(out_history_02, model_str+"_pruned_" + str(prune) + "+" + str(prune_02))
    save_in_history(in_history_02, model_str+"_pruned_" + str(prune) + "+" + str(prune_02))

    # print("=> evaluate pruned model 02")
    # validate(val_loader, pruned_model_02, criterion, args, device)

    # finetune model
    tuned_model = deepcopy(pruned_model_02)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(1):
        train(val_loader, tuned_model, 1, device, args, optimizer, criterion)
        scheduler.step()

    save_model(tuned_model, model_str+"_tuned_" + str(prune), pruned=True)

    # pruned_model = load_model("resnet50_pruned_" + str(prune), pruned=True)
    # out_history = load_out_history("resnet50_pruned_" + str(prune))
    # in_history = load_in_history("resnet50_pruned_" + str(prune))
    # # tuned_model = load_model("resnet50_tuned_" + str(prune), pruned=True)
    # # tuned_model = tuned_model.to(device)

    # compare_models(pruned_model, tuned_model)
    print("=> evaluate pruned/tuned model")
    validate(val_loader, tuned_model, criterion, args, device)

    rebuilt_model_01 = rebuild_model(pruned_model_02, pruned_model_01, verification_model, device, out_history_02, in_history_02)
    rebuilt_model_01 = rebuilt_model_01.to(device)
    compare_models(pruned_model_01, rebuilt_model_01)

    save_model(rebuilt_model_01, model_str+"_rebuilt_01_"+str(prune))

    print("=> evaluate rebuilt model 01 (no fine-tuning)")
    validate(val_loader, rebuilt_model_01, criterion, args, device)

    # print("=> Starting fine-tuning of rebuilt model 01 (Only train tune pruned channels)...")
    # tune(val_loader, rebuilt_model_01, criterion, out_history, in_history, device, args)
    #
    # print("=> evaluate rebuilt/tuned 01")
    # validate(val_loader, rebuilt_model_01, criterion, args, device)

    rebuilt_model_02 = rebuild_model(rebuilt_model_01, original_model, verification_model, device, out_history, in_history)
    rebuilt_model_02 = rebuilt_model_02 .to(device)

    print("=> evaluate rebuilt model 02 (no fine-tuning)")
    validate(val_loader, rebuilt_model_02, criterion, args, device)

    # print("=> Starting fine-tuning of rebuilt model (Only train tune pruned channels)...")
    # tune(val_loader, rebuilt_model_02, criterion, out_history, in_history, device, args)
    #
    # print("=> evaluate rebuilt/tuned 02")
    # validate(val_loader, rebuilt_model_02, criterion, args, device)
    # save_model(rebuilt_model_02, model_str+"_rebuilt_02_" + str(prune))

if __name__ == '__main__':
    main()
