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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

from misc import weights_init, save_individual_images
from models import Generator
from models import Discriminator
from models import get_classifier
from models import mahalanobis_loss

from train_and_eval import train, eval

from itertools import cycle

import pretty_errors

# Set random seed for reproducibility
manualSeed = 999
batch_size = 32
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5
ngpu = 4
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot_fake = "generated_images//mnist_to_svhn//"
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
])


transform_rbg=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_src_train = datasets.MNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True)

dataset_tgt_train = datasets.SVHN(root='./data',
                              split='test',
                              transform=transform_rbg,
                              download=True)

dataset_fake = datasets.ImageFolder(root=dataroot_fake,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataset_src_test = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform,
                              download=True)

dataset_tgt_test = datasets.SVHN(root='./data',
                              split='test',
                              transform=transform_rbg,
                              download=True)


dataloader_src_train = torch.utils.data.DataLoader(dataset_src_train, batch_size=batch_size,
                                         shuffle=True)
dataloader_src_test = torch.utils.data.DataLoader(dataset_src_test, batch_size=batch_size,
                                         shuffle=True)
dataloader_tgt_train = torch.utils.data.DataLoader(dataset_tgt_train, batch_size=batch_size,
                                         shuffle=True)
dataloader_tgt_test = torch.utils.data.DataLoader(dataset_tgt_test, batch_size=batch_size,
                                         shuffle=True)
dataloader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=batch_size,
                                         shuffle=True)

dataset_src_tgt_train = ConcatDataset((dataset_src_train, dataset_tgt_train))
dataset_src_tgt_fake_train = ConcatDataset((dataset_src_train, dataset_tgt_train, dataset_fake))

src_tgt_train_loader = torch.utils.data.DataLoader(dataset_src_tgt_train, batch_size=batch_size, shuffle=True)
src_tgt_fake_train_loader = torch.utils.data.DataLoader(dataset_src_tgt_fake_train, batch_size=batch_size, shuffle=True)

f = get_classifier('googlenet')

classifier = f.cuda()
classifier = train(classifier, src_tgt_fake_train_loader)

acc_src = eval(classifier, dataloader_src_test)
acc_tgt = eval(classifier, dataloader_tgt_test)

print('src acc {}, target acc {}'.format(acc_src, acc_tgt))
