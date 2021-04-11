import sys
import torch.optim as optim
import matplotlib.pyplot as plt

import discriminator
import generator
import utils
from params import *


def train_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250,
              batch_size=batch_size, noise_size=NOISE_DIM, num_epochs=10):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    for epoch in range(num_epochs):
        print("Starting Epoch " + str(epoch))

        for x, _ in dataloader:
            if len(x) != batch_size:
                continue

            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2 * (real_data - 0.5)).type(dtype)

            g_fake_seed = utils.sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 3, 218, 178))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = utils.sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 3, 218, 178))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()

            if iter_count % show_every == 0:
                print('\nIter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                utils.show_images(imgs_numpy[0:16], title="Generated after " + str(iter_count) + " iters")
                plt.show()
            else:
                sys.stdout.write("\r ⤑ running iter " + str(iter_count))
                sys.stdout.flush()

            iter_count += 1


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer


if __name__ == "__main__":
    dataloader, device = utils.get_data_loader()

    D = discriminator.discriminator().type(dtype)
    G = generator.cnn_generator().type(dtype)

    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)

    train_gan(D, G, D_solver, G_solver, discriminator.ls_loss, generator.ls_loss, show_every=25)
