
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.optim as optim

from generator import Generator
from discriminator import Discriminator
import utils
from params import *

"""
Code is based on the tutorial at:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    authored by Nathan Inkawhich (https://github.com/inkawhich)
"""

def training_loop(start_epoch=0, end_epoch=num_epochs):
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    iters = 0

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, end_epoch+1):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, end_epoch, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == end_epoch-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(fake)

            iters += 1
        # Save checkpoint for each epoch
        utils.save_checkpoint(epoch, netD, netG, optimizerD, optimizerG)
    for i in range(5):
        utils.show_images(img_list[-i])

def get_models():
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(utils.weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(utils.weights_init)

    return netD, netG

def plot_for_checkpoint(checkpoint_name, device, ngpu=1):
    """Displays 64 images generated for the checkpoint with name checkpoint_name.pt"""
    last_epoch, netD, netG, optD, optG = utils.load_checkpoint(f"checkpoints/{checkpoint_name}.pt", ngpu, device)
    with torch.no_grad():
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        fake = netG(fixed_noise).detach().cpu()
        utils.show_images(fake)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    dataloader, device = utils.get_data_loader()
    #utils.show_images(next(iter(dataloader))[0], "Sample Training Images")
    # Get generator and discriminator
    netD, netG = get_models()
    # Setup Adam optimizers for both G and D
    optimizerD = netD.get_optimizer()
    #last_epoch, netD, netG, optimizerD, optimizerG = utils.load_checkpoint("checkpoints/checkpoint0.pt", ngpu, device)
    optimizerG = netG.get_optimizer()
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Start training
    training_loop()
    # for i in range(5,6):
    #     plot_for_checkpoint(f"checkpoint{i}", device)
    # Display results
    #plot_losses()
