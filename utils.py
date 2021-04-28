import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch.nn as nn
from params import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from discriminator import CondDiscriminator
from generator import CondGenerator

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

"""
get_data_loader() and weights_init() based on the tutorial at:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    authored by Nathan Inkawhich (https://github.com/inkawhich)
"""

def get_data_loader():
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.CelebA(root=dataroot,
                            split="all",
                            download=False,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return dataloader, device

def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 1)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(epoch, netD, netG, optD, optG, directory="checkpoints"):
    # Create directory if it doesn't exist
    if not os.path.isdir(directory):
        os.makedirs(directory)
    torch.save({
        "epoch": epoch,
        "netD_state_dict": netD.state_dict(),
        "netG_state_dict": netG.state_dict(),
        "optD_state_dict": optD.state_dict(),
        "optG_state_dict": optG.state_dict()
    }, f"{directory}/checkpoint{epoch}.pt")

def load_checkpoint(path, ngpu, device):
    checkpoint_dict = torch.load(path)
    epoch = checkpoint_dict["epoch"]
    # Load models
    netD = CondDiscriminator(ngpu).to(device)
    netD.load_state_dict(checkpoint_dict["netD_state_dict"])
    netG = CondGenerator(ngpu).to(device)
    netG.load_state_dict(checkpoint_dict["netG_state_dict"])

    # Load optimizers
    optD = netD.get_optimizer()
    optG = netG.get_optimizer()
    optD.load_state_dict(checkpoint_dict["optD_state_dict"])
    optG.load_state_dict(checkpoint_dict["optG_state_dict"])
    return epoch, netD, netG, optD, optG

def show_images(images, title=""):
    # Convert images from [-1, 1] to [0, 1] range
    images = (images + 1) / 2
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0, hspace=0.05)

    if title != "":
        fig.suptitle(title)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(np.transpose(img.reshape([3, image_size, image_size]), (1, 2, 0)))

    plt.axis('off')
    plt.show()
