import torch.nn as nn
from torch.nn import init
from utils import Flatten, Unflatten
from params import *


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavieruniform(m.weight.data)


def generator(noise_dim=NOISE_DIM):
    return nn.Sequential(
         nn.Linear(noise_dim, 1024),
         nn.ReLU(),
         nn.Linear(1024, 1024),
         nn.ReLU(),
         nn.Linear(1024, noise_dim),
         nn.Sigmoid())


def cnn_generator(noise_dim=NOISE_DIM):
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 6272),
        nn.ReLU(),
        nn.BatchNorm1d(6272),
        Unflatten(batch_size, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        Flatten(),
        nn.Linear(784, 3 * 218 * 178),
        nn.Sigmoid())


def ls_loss(scores_fake):
    loss = torch.mean(torch.pow(scores_fake - torch.ones(scores_fake.size()[0]).type(dtype), 2)) / 2
    return loss
