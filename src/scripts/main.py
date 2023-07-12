import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import argparse

# import local libraries
from metrics import *


def switch(model):
    if model == "resnet50":
        return torchvision.models.resnet50(pretrained=True)
    elif model == "resnet18":
        return torchvision.models.resnet18(pretrained=True)
    elif model == "vgg":
        return torchvision.models.vgg19(pretrained=True)

if __name__ == '__main__':

    ngpus_per_node = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--out', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--prune', type=float)

    args = parser.parse_args()
    epochs = args.epoch
    output_path = args.out
    model = switch(args.model)
    prune_amount = args.prune

    print("Received from args:", epochs, output_path, model.__module__, str(prune_amount))

    transformations = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    
    run_identifier = args.model+'_epochs_'+str(epochs)+'_prune_'+str(prune_amount)+"_"

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using CUDA toolkit' if torch.cuda.is_available() else 'Using cpu :L')

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    # acc_top1 = validate(model)
    

    # pruned_model = apply_prune(model, prune_amount)
    # pruned_acc_top1, metrics = validate(pruned_model)

    # save_checkpoint(model.state_dict(), output_path, run_identifier)

    # with open(output_path+run_identifier+'top1_accuracy.txt', 'w') as f:
    #     f.write(str(pruned_acc_top1))

    # with open(output_path+run_identifier+'metrics.txt', 'w') as f:
    #     f.write(metrics)

    # # fine-tuning here
    # tuned_model = tune(model)
    # tuned_acc_top1, metrics = validate(tuned_model)
    #
    # save_checkpoint(tuned_model.state_dict(), output_path, run_identifier)
    # print(("Training done"))

    # test_model = tune(model)






