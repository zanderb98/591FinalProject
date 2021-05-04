import torch
import torch.nn as nn
import torch.optim as optim

"""
Code is based on the tutorial at:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    authored by Nathan Inkawhich (https://github.com/inkawhich)
"""

class CondDiscriminator(nn.Module):
    """Implementation of DCGAN Discriminator, conditioned on annotations from CelebA dataset."""
    def __init__(self, ndf):
        super(CondDiscriminator, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.image_size = 64 # Network structure is coupled with image size
        self.annot_embedding = nn.Linear(40, self.image_size * self.image_size, bias=False)
        self.block1 = nn.Sequential()
        self.main = nn.Sequential(
            # input is (3 + 1) x 64 x 64
            nn.Conv2d(3 + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
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

    def get_optimizer(self):
        return optim.AdamW(self.parameters(), lr=0.0002*0.8, betas=(0.5, 0.999))

    def forward(self, x, annot):
        annot = self.softmax(annot)  # Ensure annotations add to one
        annot = self.annot_embedding(annot) # Return average of embeddings for annotations that equal 1
        annot = torch.reshape(annot, (x.size(0), 1, self.image_size, self.image_size))
        return self.main(torch.cat((x, annot), dim=1)) # Concatenate embedding as another feature map