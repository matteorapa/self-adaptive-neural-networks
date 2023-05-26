import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from methods import get_external_requirements
from torchvision.models import ResNet


def get_external_requirements():
    return 0.1


def apply_prune(model: ResNet) -> ResNet:
    for name, module in model.named_modules():
        # prune connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):

            prune.ln_structured(module, name="weight", amount=get_external_requirements(), n=2, dim=0)
        # prune connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=get_external_requirements(), n=2, dim=0)

    prune.remove(module, 'weight')

    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )

    torch.save({'state_dict': model.state_dict()}, 'pruned.pth')
    return model



