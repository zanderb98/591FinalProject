import torch
import torch.nn as nn
import torch.optim as optim

"""
Code is based on the tutorial at:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    authored by Nathan Inkawhich (https://github.com/inkawhich)
"""

class CondGenerator(nn.Module):
    """Implementation of DCGAN Generator, conditioned on annotations from CelebA dataset."""
    def __init__(self, ngf, nz):
        super(CondGenerator, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.embed_dims = 40
        self.annot_embedding = nn.Linear(40, self.embed_dims, bias=False)
        self.main = nn.Sequential(
            # input is Z and embedding, going into a convolution
            nn.ConvTranspose2d(nz + self.embed_dims, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )
        # Initialize weights
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 1.0) # Match distribution of embedding to latent vector

    def get_optimizer(self):
        return optim.AdamW(self.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, z, annot):
        annot = self.softmax(annot) # Ensure annotations add to one
        annot = self.annot_embedding(annot) # Return average of embeddings for annotations that equal 1
        annot = torch.reshape(annot, (z.size(0), self.embed_dims, 1, 1))
        return self.main(torch.cat((z, annot), dim=1)) # Concatenate embedding to latent vector and apply regular DCGAN generator