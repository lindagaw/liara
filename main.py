from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from misc import weights_init, save_individual_images
from models import Generator
from models import Discriminator, Classifier
from models import mahalanobis_loss

from train_and_eval import train

from itertools import cycle

import pretty_errors

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot_src = "datasets//office-31-intact//amazon//images//"
dataroot_tgt = "datasets//office-31-intact//dslr//images//"
#dataroot = "celebs//"

# Batch size during training
batch_size = 128

image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 1500
lr = 0.00002
beta1 = 0.5
ngpu = 4

dataset_src = dset.ImageFolder(root=dataroot_src,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset_tgt = dset.ImageFolder(root=dataroot_tgt,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader_src = torch.utils.data.DataLoader(dataset_src, batch_size=batch_size,
                                         shuffle=True)
dataloader_tgt = torch.utils.data.DataLoader(dataset_tgt, batch_size=batch_size,
                                         shuffle=True)

classifier = Classifier(ngpu).cuda()
classifier = train(classifier, dataloader_src)
