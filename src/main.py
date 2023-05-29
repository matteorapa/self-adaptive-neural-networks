import pandas as pd
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torchvision.models import ResNet
import torchvision.transforms as transforms
import argparse
import tarfile
from pathlib import Path
import glob
import csv
import os
import json
import io
import pickle
import PIL.Image
import zipfile
import tarfile

# import local libraries
from methods import *
from prune import *
# from finetune import *
from metrics import *
from paths import *

def switch(model):
    if model == "resnet50":
        return torchvision.models.resnet50(pretrained=True)
    elif model == "resnet18":
        return torchvision.models.resnet18(pretrained=True)
    elif model == "vgg":
        return torchvision.models.vgg19(pretrained=True)

class TARDataset(torch.utils.data.Dataset):
    """Dataset for packed ZIP to pass to a PyTorch dataloader

    Args:
        path (str): location where the images are stored on disk
        transform (obj): torchvision.transforms object or None
        load_encoded (bool): whether the images within the .tar file are encoded or saved as bytes directly
        label_file (str): path to save the cached getmembers() output as this may take a while for larger dataset
    Returns:
        torch Dataset: to pass to a dataloader
    """

    def __init__(self, path, transform=None, load_encoded=False, label_file=None):
        print(f"Initializing TAR dataset for: {path}")
        self.path = path
        self.transform = transform
        self.load_encoded = load_encoded

        # First uncompress because .gz cannot be read in parallel
        if self.path.endswith(".tar.gz"):
            print(
                "WARNING: tar file is compressed which drastically impacts performance -> gunzip -k <file>"
            )

        # TAR is not thread-safe so give file handle to each worker
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        self.tar_handle = {worker: tarfile.open(path)}

        # Store headers of all files and folders by name
        # self.members = sorted(self.tar_handle[worker].getmembers(), key=lambda m: m.name)

        # store headers of all files and folders by name
        if label_file:
            with open(label_file, "rb") as fp:
                self.members_by_name = pickle.load(fp)

        else:
            # get.members() takes very long for larger TAR archives so cache the members in a byte file
            self.members_by_name = {
                m.name: m
                for m in sorted(
                    self.tar_handle[worker].getmembers(), key=lambda m: m.name
                )
            }
            with open(os.path.join(Path(path).parent, "members"), "wb") as fp:
                pickle.dump(self.members_by_name, fp)
            print(
                f"Finished create a members file. Please add the following path to the init as label_file next time: ",
                str(Path(path).parent) + "members",
            )

        self._get_all_samples()
        label_fname = None  # or "String"

        if len(self.label_fname) >= 1 and label_fname is not None:
            self._parse_label_file(worker, label_fname)
        else:
            self._get_filler_labels()

    def _get_all_samples(self, label_exts=(".json")):
        """Sort labels and images from the names
        Args:
            label_exts: the file extensions which will be categorized as labels
        Returns:
            None: filled lists of self.samples and self.label_fname
        """
        self.members_by_name = {m.name: m for m in self.members}
        self.samples = []
        self.label_fname = []
        PIL.Image.init()

        for m in self.members_by_name.values():
            m_name = m.name
            if m_name.lower().endswith(tuple(PIL.Image.EXTENSION.keys())):
                self.samples.append(m_name)
            elif m_name.lower().endswith(label_exts):
                self.label_fname.append(m_name)
        self.file_ext = os.path.splitext(self.samples[0])[1]

        self.length = len(self.samples)

    def _parse_label_file(self, worker, label_fname=""):
        """Sort labels and images from the names
        Args:
            worker (int): worker id
            label_fname (str): hardcoded label file name
        Returns:
            None: parsed labels
        """
        # If no hardcoded label filename is given, use the first one from the list by _get_all_samples()
        if label_fname == "":
            if len(self.label_fname) != 1:
                print(
                    f"WARNING: found {len(self.label_fname)} label files - using only the first"
                )

            label_fname = self.label_fname[0]

        # Parse labels from json file
        label_file = self.tar_handle[worker].extractfile(label_fname)
        labels = json.load(label_file)["labels"]
        labels = dict(sorted(labels, key=lambda item: item[1]))

        labels = [labels[fname.replace("\\", "/")] for fname in self.samples]
        self.labels = np.array(labels, dtype=np.uint8)

    def _get_filler_labels(self):
        # Placeholder for datasets with no labels like FFHQ
        self.labels = [0] * self.length

    def _get_file(self, fname):
        """Retrieve file handle for a given file name within the ZIP file"""
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_handle:
            self.tar_handle[worker] = tarfile.open(self.path)

        return self.tar_handle[worker].extractfile(self.members_by_name[fname])

    def _get_image(self, fname):
        fname = self._get_file(fname)
        jpgs = (".jpg", ".jpeg")
        if not self.load_encoded:
            # In case of having the image saved as bytes:
            image = np.frombuffer(fname.read(), dtype=np.uint8).reshape(256, 256, 3)
        else:
            if self.file_ext.lower() == ".png" and pyspng:
                image = pyspng.load(fname.read())
            elif self.file_ext.lower() in jpgs and turbojpeg:
                try:
                    image = turbojpeg_decoder.decode(fname.read(), pixel_format=0)
                except IOError:  # Catch jpgs which are actually encoded as PNG
                    image = PIL.Image.open(fname).convert("RGB")
            else:
                image = PIL.Image.open(fname.read()).convert("RGB")

        return image

    def _get_label(self, index):
        return self.labels[index]

    def __getitem__(self, index):
        fname_image = self.samples[index]
        image = self._get_image(fname_image)
        label = self._get_label(index)

        if self.transform:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        return image, label

    def __len__(self):
        return self.length

    def __del__(self):
        """Clean all file handles of the workers on exit"""
        if hasattr(self, "tar_handle"):
            for o in self.tar_handle.values():
                o.close()

    def __getstate__(self):
        """Serialize without the ZipFile references, for multiprocessing compatibility"""
        state = dict(self.__dict__)
        state["tar_handle"] = {}
        return state
def tune(model: ResNet) -> ResNet:
    global best_prec1

    # model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # print("=> loading checkpoint '{}'".format(args.resume))
    # checkpoint = torch.load(args.resume)
    # args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print("=> loaded checkpoint '{}' (epoch {})"
    # .format(args.resume, checkpoint['epoch']))

    tar_train_dataset = TARDataset(path=TRAIN_DIR, transform=transformations)
    cudnn.benchmark = True

    # train_dataset = datasets.ImageFolder(
    #     TRAIN_DIR,
    #     transformations)

    train_loader = torch.utils.data.DataLoader(
        tar_train_dataset, batch_size=16, shuffle=True)

    history_score = np.zeros((epochs + 1, 1))
    np.savetxt('record.txt', history_score, fmt = '%10.5f', delimiter=',')
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(model)
        history_score[epoch] = prec1
        np.savetxt('record.txt', history_score, fmt = '%10.5f', delimiter=',')

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, "thepath")

    history_score[-1] = best_prec1
    np.savetxt('record.txt', history_score, fmt = '%10.5f', delimiter=',')
    return model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    total_correct = 0
    total_images = 0
    i = 0

    model.cuda()
    # switch to train mode
    model.train()

    end = time.time()
    print("Starting training...")


    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, target = images.cuda(), target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        _, predicted = torch.max(output.data, 1)
        total_images += target.size(0)
        total_correct += (predicted == target).sum().item()
        prec1 = 100 * total_correct / total_images

        losses.update(loss, target.size(0))
        top1.update(prec1, target.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(top1.avg)
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Prec@1 {top1.avg:.3f}\t'
            #       .format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, loss=losses, top1=top1))
def save_checkpoint(state, filepath, name):
    torch.save(state, os.path.join(filepath, name+'checkpoint.pth'))
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(model):
    # Set the batch size for validation
    batch_size = 64

    # Define the transform for the validation data
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the validation data
    val_data = torchvision.datasets.ImageFolder(VALIDATION_DIR, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Set the model to evaluate mode
    model.eval()

    # Define the criterion for validation
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize the variables for validation
    total_correct = 0
    total_images = 0
    i = 0

    # Validate the model on the validation set
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_images += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    i += 1
                    
    # Compute the accuracy and print the result
    accuracy = 100 * total_correct / total_images
    print('Accuracy on the validation set: {:.2f}%'.format(accuracy))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return accuracy, str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == '__main__':

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
    # acc_top1 = validate(model)
    

    # pruned_model = apply_prune(model, prune_amount)
    # pruned_acc_top1, metrics = validate(pruned_model)

    # save_checkpoint(model.state_dict(), output_path, run_identifier)

    # with open(output_path+run_identifier+'top1_accuracy.txt', 'w') as f:
    #     f.write(str(pruned_acc_top1))

    # with open(output_path+run_identifier+'metrics.txt', 'w') as f:
    #     f.write(metrics)

    # # fine-tuning here
    tuned_model = tune(model)
    tuned_acc_top1, metrics = validate(tuned_model)

    save_checkpoint(tuned_model.state_dict(), output_path, run_identifier)

    # test_model = tune(model)






