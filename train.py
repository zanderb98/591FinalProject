
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.optim as optim

from generator import CondGenerator
from discriminator import CondDiscriminator
import utils
from params import *

"""
Code is based on the tutorial at:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    authored by Nathan Inkawhich (https://github.com/inkawhich)
"""

def filtered_annots(annot, indices=(4,5,8,9,11,16,17,18,20,22,24,25,26,28,32,33,35,39)):
    return annot[:,indices]

def training_loop(start_epoch=0, end_epoch=num_epochs):
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    iters = 0

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fixed_annot = next(iter(dataloader))[1].type(torch.float).to(device)
    fixed_annot = filtered_annots(fixed_annot)
    #fixed_annot = torch.randint(low=0,high=2,size=(64,40),device=device, dtype=torch.float)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, end_epoch+1):
        D_x_minus_D_G_z = 0
        alpha = (1 / 250)
        lr_mult = 1.5 ** alpha
        print(f"lr_mult: {lr_mult}")
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            real_annot = data[1].type(torch.float).to(device)
            real_annot = filtered_annots(real_annot)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu, real_annot).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            #fake_annot = torch.randint(low=0,high=2,size=(b_size,40),device=device, dtype=torch.float)
            fake_annot = real_annot # Avoid having annotations that would never actually occur
            # Generate fake image batch with G
            fake = netG(noise, fake_annot)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), fake_annot).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Modify stats
            D_x_minus_D_G_z = D_x_minus_D_G_z * (1 - alpha) + alpha * (D_x - D_G_z1)

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D, healthy range of 0.2 to 0.4 higher for true vs fake
            # if D_x - D_G_z1 > 0.4 and optimizerD.param_groups[0]['lr'] > 0.25 * lr: # Winning, decrease learning rate
            #     optimizerD.param_groups[0]['lr'] /= lr_mult
            # elif D_x - D_G_z1 < 0.2 and optimizerD.param_groups[0]['lr'] < lr:
            #     optimizerD.param_groups[0]['lr'] *= lr_mult
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, fake_annot).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            # if D_G_z1_mov_avg < 0.5 and optimizerG.param_groups[0]['lr'] < lr: # Losing, increase learning rate
            #     optimizerG.param_groups[0]['lr'] *= 1.003
            # elif optimizerG.param_groups[0]['lr'] > 0.25 * lr:
            #     optimizerG.param_groups[0]['lr'] /= 1.003
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tD_lr: %.8f'
                    % (epoch, end_epoch, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, optimizerD.param_groups[0]['lr']))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == end_epoch-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise, fixed_annot).detach().cpu()
                img_list.append(fake)

            iters += 1
        # Save checkpoint for each epoch
        utils.save_checkpoint(epoch, netD, netG, optimizerD, optimizerG)
    for i in range(len(img_list) - 5, len(img_list)):
        utils.show_images(img_list[i])

def get_models():
    # Create the generator
    netG = CondGenerator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(utils.weights_init)
    netG.apply(utils.init_linear)

    # Create the Discriminator
    netD = CondDiscriminator(ngpu).to(device)

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
        fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
        #fixed_annot = torch.randint(low=0,high=2,size=(batch_size,40),device=device).type(torch.float)
        fixed_annot = next(iter(dataloader))[1].type(torch.float).to(device)
        fake = netG(fixed_noise, fixed_annot).detach().cpu()
        utils.show_images(fake)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    dataloader, device = utils.get_data_loader()
    #utils.show_images(next(iter(dataloader))[0], "Sample Training Images")
    # Get generator and discriminator
    netD, netG = get_models()
    #last_epoch, netD, netG, optimizerD, optimizerG = utils.load_checkpoint("checkpoints/checkpoint4.pt", ngpu, device)
    # optimizerD.param_groups[0]['lr'] *= 0.25
    # optimizerD.param_groups[0]['weight_decay'] = 0.5
    # Setup Adam optimizers for both G and D
    optimizerD = netD.get_optimizer()
    optimizerG = netG.get_optimizer()
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Start training
    training_loop(start_epoch=0, end_epoch=4)
    # for i in range(5):
    #     plot_for_checkpoint(f"checkpoint{i}", device)
    # Display results
    #plot_losses()
