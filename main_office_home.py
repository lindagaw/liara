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
from models import balance

from train_and_eval import train, eval

from itertools import cycle

import pretty_errors
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 32
image_size = 299

# Art, Clipart  Product  Real World
dataroot_art = "datasets//OfficeHome//Art//"
dataroot_clipart = "datasets//OfficeHome//Clipart//"
dataroot_product = "datasets//OfficeHome//Product//"
dataroot_realworld = "datasets//OfficeHome//Real World//"

transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #AddGaussianNoise(0., 1.)
])

dataset_art = datasets.ImageFolder(root=dataroot_art,
                           transform=transform)
dataset_clipart = datasets.ImageFolder(root=dataroot_clipart,
                           transform=transform)
dataset_product = datasets.ImageFolder(root=dataroot_product,
                           transform=transform)
dataset_realworld = datasets.ImageFolder(root=dataroot_realworld,
                           transform=transform)

train_set_art, test_set_art = torch.utils.data.random_split(dataset_art, [int(len(dataset_art)*0.8), len(dataset_art)-int(len(dataset_art)*0.8)])
train_set_clipart, test_set_clipart = torch.utils.data.random_split(dataset_clipart, [int(len(dataset_clipart)*0.8), len(dataset_clipart)-int(len(dataset_clipart)*0.8)])
train_set_product, test_set_product = torch.utils.data.random_split(dataset_product, [int(len(dataset_product)*0.8), len(dataset_product)-int(len(dataset_product)*0.8)])
train_set_realworld, test_set_realworld = torch.utils.data.random_split(dataset_realworld, [int(len(dataset_realworld)*0.8), len(dataset_realworld)-int(len(dataset_realworld)*0.8)])

#dataset_art_product = datasets.ImageFolder(root='generated_images//office_home_art_to_product//',
#                           transform=transform)
#dataset_art_clipart = datasets.ImageFolder(root='generated_images//office_home_clipart_to_product',
#                           transform=transform)
#dataset_clipart_product = datasets.ImageFolder(root='generated_images//office_home_clipart_to_product',
#                           transform=transform)


train_art_clipart = ConcatDataset((train_set_art, train_set_clipart))
train_art_product = ConcatDataset((train_set_art, train_set_product))
train_art_realworld = ConcatDataset((train_set_art, train_set_realworld))
train_product_clipart = ConcatDataset((train_set_product, train_set_clipart))
train_product_realworld = ConcatDataset((train_set_product, train_set_realworld))
train_clipart_realworld = ConcatDataset((train_set_clipart, train_set_realworld))

dataloader_train_art_clipart = torch.utils.data.DataLoader(train_art_clipart, batch_size=batch_size, shuffle=True)
dataloader_train_art_product = torch.utils.data.DataLoader(train_art_product, batch_size=batch_size, shuffle=True)
dataloader_train_art_realworld = torch.utils.data.DataLoader(train_art_realworld, batch_size=batch_size, shuffle=True)
dataloader_train_product_clipart = torch.utils.data.DataLoader(train_product_clipart, batch_size=batch_size, shuffle=True)
dataloader_train_product_realworld = torch.utils.data.DataLoader(train_product_realworld, batch_size=batch_size, shuffle=True)
dataloader_train_clipart_realworld = torch.utils.data.DataLoader(train_clipart_realworld, batch_size=batch_size, shuffle=True)

dataloader_test_art = torch.utils.data.DataLoader(test_set_art, batch_size=batch_size, shuffle=True)
dataloader_test_clipart = torch.utils.data.DataLoader(test_set_clipart, batch_size=batch_size, shuffle=True)
dataloader_test_product = torch.utils.data.DataLoader(test_set_product, batch_size=batch_size, shuffle=True)
dataloader_test_realworld = torch.utils.data.DataLoader(test_set_realworld, batch_size=batch_size, shuffle=True)

f = get_classifier('inception_v3', pretrain=True)

print(f)

f.fc = nn.Linear(2048, 65)

classifier = f.cuda()

#classifier = train(classifier, dataloader_train_art_product)
#classifier = train(classifier, dataloader_train_art_clipart)
#classifier = train(classifier, dataloader_train_art_realworld)
#classifier = train(classifier, dataloader_train_product_clipart)
#classifier = train(classifier, dataloader_train_product_realworld)
classifier = train(classifier, dataloader_train_clipart_realworld)

print('eval on clipart')
acc = eval(classifier, dataloader_test_clipart)
print('eval on realworld')
acc = eval(classifier, dataloader_test_realworld)
