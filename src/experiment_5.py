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

    print("=> using pre-trained model '{}'".format(args.arch))
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    verification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    original_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

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

    # validate(val_loader, model, criterion, args)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # validate(val_loader, verification_model, criterion, args, device)
    # pruned_model, history_steps = apply_channel_prune(model, 0.125, example_inputs)
    #
    in_channels_history_dict = get_in_channel_history(original_model, pruned_model, history_steps)
    #
    # # save model, out_channels, in_channels
    # save_model(model, "resnet50_pruned_0.125", pruned=True)
    # save_out_history(history_steps, "resnet50_pruned_0.125")
    # save_in_history(in_channels_history_dict, "resnet50_pruned_0.125")

    pruned_model = load_model("resnet50_pruned_0.125", pruned=True)

    # model_statistics = summary(pruned_model, (1, 3, 224, 224), depth=3,
    #                            col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], )
    # model_statistics_str = str(model_statistics)

    # finetune model
    # tuned_model = train(train_loader, model, criterion, optimizer, 0, device, args)
    # print("=> evaluate pruned model after tuning")
    # validate(val_loader, tuned_model, criterion, args)

    show_layers(pruned_model)
    compare_models(original_model, verification_model)
    rebuilt_model = rebuild_model(pruned_model, original_model, device)
    rebuilt_model = rebuilt_model.to(device)
    compare_models(verification_model, rebuilt_model)

    isSame = compareModelWeights(verification_model, rebuilt_model)
    if isSame:
        print("The models are the same!")
    else:
        print("The models not the same :(")

    save_onnx(verification_model, "resnet50_verif", device)
    save_onnx(rebuilt_model, "resnet50_rebuilt", device)
    save_model(rebuilt_model, "resnet50_rebuilt")

    validate(val_loader, verification_model, criterion, args, device)
    validate(val_loader, rebuilt_model, criterion, args, device)


    # if torch.equal(tuned_model.conv1.weight.data[51, :, :, :], rebuilt_model.conv1.weight.data[58, :, :, :]):
    #     print("values match")
    #
    # if torch.equal(test_model.conv1.weight.data[59, :, :, :], rebuilt_model.conv1.weight.data[59, :, :, :]):
    #     print("values match for non updated idx 59")



    # # validate model after rebuilding, degraded accuracy expected
    # print("=> evaluate rebuilt model (no fine-tuning)")
    # validate(val_loader, rebuilt_model, criterion, args, device)
    # torch.save(tp.state_dict(bigger_model), "exp_3_model_resnet50_rebuilt_0.125.pth")
    #
    #
    # print("=> Starting fine-tuning of rebuilt model (Only train tune pruned channels)...")
    # for epoch in range(0, 10):
    #
    #     tune(val_loader, rebuilt_model, criterion, optimizer, epoch, device, args, step_history)
    #
    #     scheduler.step()
    #     print("=> evaluate rebuilt and tuned model")
    #     validate(val_loader, rebuilt_model, criterion, args, device)

    # torch.save(tp.state_dict(model), "exp_3_model_resnet18_rebuilt_0.3_tuned.pth")
    #
    # model_stats = summary(model, (1, 3, 224, 224), depth=3, col_names=["input_size", "output_size", "num_params", "mult_adds"])


if __name__ == '__main__':
    main()
