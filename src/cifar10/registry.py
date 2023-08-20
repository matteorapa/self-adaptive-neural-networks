from pyexpat import model
from torchvision import datasets, transforms as T
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import os, sys
import engine.models as models
import engine.utils as utils
from functools import partial
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10_224':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100_224': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}


MODEL_DICT = {
    'resnet18': models.cifar.resnet.resnet18,
    'resnet34': models.cifar.resnet.resnet34,
    'resnet50': models.cifar.resnet.resnet50,
    'mobilenetv2': models.cifar.mobilenetv2.mobilenetv2,
    'resnet20': models.cifar.resnet_tiny.resnet20,
    'resnet32': models.cifar.resnet_tiny.resnet32,
    'resnet44': models.cifar.resnet_tiny.resnet44,
    'resnet56': models.cifar.resnet_tiny.resnet56,
    'resnet110': models.cifar.resnet_tiny.resnet110,
    'resnext50': models.cifar.resnext.resnext50,
    'resnext101': models.cifar.resnext.resnext101,
    'resnext152': models.cifar.resnext.resnext152,
}

IMAGENET_MODEL_DICT={
    "resnet50": models.imagenet.resnet50,
    "mobilenet_v2": models.imagenet.mobilenet_v2,
}


def get_model(name: str, num_classes, pretrained=False, target_dataset='cifar', **kwargs):
    if target_dataset == "imagenet":
        
        model = IMAGENET_MODEL_DICT[name](pretrained=pretrained)
    elif 'cifar' in target_dataset:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'cifar10' )
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
        input_size = (1, 3, 32, 32)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, input_size, train_transform, val_transform
    return num_classes, train_dst, val_dst, input_size

