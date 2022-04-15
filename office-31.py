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
image_size = 224

dataroot = "datasets//office-31-intact//webcam//images//"

transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #AddGaussianNoise(0., 1.)
])

dataset = datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])


dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

f = get_classifier('inception_v3', pretrain=True)

print(f)

f.fc = nn.Linear(2048, 31)

classifier = f.cuda()
classifier = train(classifier, dataloader_train, dataloader_test)

acc = eval(classifier, dataloader_test)

print(acc)
