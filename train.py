import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from generator import CondGenerator
from discriminator import CondDiscriminator
import utils

"""
Code is based on the tutorial at:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    authored by Nathan Inkawhich (https://github.com/inkawhich)
"""

# Constants
image_size = 64
batch_size = 128
workers = 2
latent_dims = 100
ngf = 64 # Parameter for number of feature maps in generator
ndf = 64 # Parameter for number of feature maps in discriminator

def get_data_loader():
    dataset = dset.CelebA(root=args.celeba_loc,
                            split="all",
                            download=False,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

def train(args):
    # Get device and dataloader
    device = torch.device(args.device)
    dataloader = get_data_loader()

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Load checkpoint if needed
    start_epoch = 1
    if args.resume_from is not None:
        start_epoch = args.resume_from + 1
        epoch, D, G, optimizerD, optimizerG = utils.load_checkpoint(f"checkpoints/checkpoint{args.resume_from}.pt", device,ndf=ndf,ngf=ngf,latent_dims=latent_dims)
        print(f"Resuming from checkpoint after epoch: {args.resume_from}")
    else: # Get new networks and optimizers
        G = CondGenerator(ngf, latent_dims).to(device)
        D = CondDiscriminator(ndf).to(device)
        optimizerG = G.get_optimizer()
        optimizerD = D.get_optimizer()

    print("Starting training loop...")
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        for i, data in enumerate(dataloader):
            # Format batch
            real_imgs = data[0].to(device) # Batch of training images
            real_annots = data[1].type(torch.float).to(device) # 40 annotations for each image
            curr_batch_size = real_imgs.size(0) # Number of training images in this batch
            labels = torch.full((curr_batch_size,), 1, dtype=torch.float, device=device) # Indicate real as 1 label

            # Get gradient of D for real
            D.zero_grad() # Initialize the gradient of D to 0
            output = D(real_imgs, real_annots).view(-1) # Predict real or fake for real images
            D_x = output.mean().item() # Portion of real images correctly labelled
            errD_real = criterion(output, labels)
            errD_real.backward() # Backpropogate loss for D over real images

            # Get gradient of D for fake
            latent_vector = torch.randn(curr_batch_size, latent_dims, 1, 1, device=device) # Generate noise
            fake_imgs = G(latent_vector, real_annots) # Use real annotations, randomly generated annotations aren't realistic
            output = D(fake_imgs.detach(), real_annots).view(-1) # Predict real or fake for fake images
            D_G_z1 = output.mean().item() # Portion of fake images correctly labelled
            labels.fill_(0) # Indicate false as 0 label
            errD_fake = criterion(output, labels)
            errD_fake.backward() # Backpropogate loss for D over fake images

            # Make gradient step for D
            optimizerD.step()

            # Update G
            G.zero_grad() # Initialize the gradient of G to 0
            labels.fill_(1) # Generator wants discriminator to yield 1 for fake images
            output = D(fake_imgs, real_annots).view(-1) # Predict real or fake for fake images
            D_G_z2 = output.mean().item() # Portion of fake images correctly labelled, after step
            errG = criterion(output, labels)
            errG.backward() # Backpropogate loss for G over fake images
            optimizerG.step() # Make gradient step

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, start_epoch+args.num_epochs-1, i, len(dataloader),
                        (errD_fake+errD_real).item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save checkpoint for each epoch
        utils.save_checkpoint(epoch, D, G, optimizerD, optimizerG)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional DCGAN')
    
    parser.add_argument("--celeba_loc", default="./images", type=str, help="Directory of CelebA dataset.")
    parser.add_argument("--display", default=-1, type=int, help="Epoch of checkpoint to show batch of images for.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="'cuda:index' or 'cpu'")
    parser.add_argument("--resume_from", type=int, help="Epoch of checkpoint to resume training from.")
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to continue training.")

    args = parser.parse_args() # Get command-line arguments
    if args.display >= 0:
        # Display a batch of images
        device = torch.device(args.device)
        epoch, D, G, optimizerD, optimizerG = utils.load_checkpoint(f"checkpoints/checkpoint{args.display}.pt", device,ndf=ndf,ngf=ngf,latent_dims=latent_dims)
        dataloader = get_data_loader()
        with torch.no_grad():
            real_annot = next(iter(dataloader))[1].type(torch.float).to(device) # Retrieve some real annotions
            latent_vector = torch.randn(batch_size, latent_dims, 1, 1, device=device) # Generate noise
            fake_imgs = G(latent_vector, real_annot).detach().cpu() # Evaluate generated images
            utils.show_images(fake_imgs, title=f"Batch of Fake Images After {epoch} Epochs")
    else:
        print(f"Command-line args: {args}")
        train(args) # Start training loop