import torch
import os
import time
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torchvision.models import ResNet
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

from methods import *
from paths import *
from validate import *
cudnn.benchmark = True

def tune(model: ResNet, epochs, device) -> ResNet:
    global best_prec1

    # model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # t = time.time()
    # print(f'Done in {float(time.time() - t):.1f} seconds.')


    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transformations)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)

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

def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    total_correct = 0
    total_images = 0
    i = 0

    model.to(device)
    # switch to train mode
    model.train()

    end = time.time()
    print("Starting training...")


    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, target = images.to(device), target.to(device)

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