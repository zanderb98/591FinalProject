import torch.nn as nn
from torch.nn import init
from utils import Flatten, Unflatten
from params import *


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavieruniform(m.weight.data)


def discriminator():
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(
         Flatten(),
         nn.Linear(178*218*3, 256),
         nn.LeakyReLU(negative_slope=0.01),
         nn.Linear(256, 256),
         nn.LeakyReLU(negative_slope=0.01),
         nn.Linear(256, 1)
    )
    return model


def cnn_discriminator():
    return nn.Sequential(
        Unflatten(N=128, C=3, H=218, W=178),
        nn.Conv2d(3, 32, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(133824, 1024),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(1024, 1))


def ls_loss(scores_real, scores_fake):
    loss = torch.mean(torch.pow(scores_real - torch.ones(scores_real.size()[0]).type(dtype), 2)) / 2 + \
           torch.mean(torch.pow(scores_fake, 2)) / 2

    return loss
