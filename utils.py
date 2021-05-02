import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision
import torch
from PIL import Image

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

from discriminator import CondDiscriminator
from generator import CondGenerator

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

def load_checkpoint(path, device, ndf=64, ngf=64, latent_dims=100):
    checkpoint_dict = torch.load(path)
    epoch = checkpoint_dict["epoch"]
    # Load models
    netD = CondDiscriminator(ndf).to(device)
    netD.load_state_dict(checkpoint_dict["netD_state_dict"])
    netG = CondGenerator(ngf, latent_dims).to(device)
    netG.load_state_dict(checkpoint_dict["netG_state_dict"])
    # Load optimizers
    optD = netD.get_optimizer()
    optG = netG.get_optimizer()
    optD.load_state_dict(checkpoint_dict["optD_state_dict"])
    optG.load_state_dict(checkpoint_dict["optG_state_dict"])
    return epoch, netD, netG, optD, optG

def write_images(images, start_index):
    """Write each image in batch to files ./generated/{i+start_index}.jpg for i in range(len(images))"""
    # Convert images from [-1, 1] to [0, 1] range
    images = (images + 1) / 2
    for i, img in enumerate(images):
        torchvision.utils.save_image(img.float(), f"./generated/{i+start_index}.jpg")

def show_images(images, title=""):
    # Convert images from [-1, 1] to [0, 1] range
    image_size = images.size(-1)
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
