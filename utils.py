import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.utils as vutils

import numpy as np

def get_dataloader():
    dataset = dset.ImageFolder(root="images",
                           transform=transforms.ToTensor()
                          )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return dataloader, device