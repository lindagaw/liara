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

from sklearn.manifold import TSNE
import seaborn as sns

from helper import weights_init, all_imgs_from_dataloader
from generator import Generator
from discriminator import Discriminator
from mahalanobis import mahalanobis_loss

from itertools import cycle

import pretty_errors

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
#dataroot = "office-31//amazon//images//"
#dataroot = "celebs//"

dataroot_amazon = "office-31//amazon//images//"
dataroot_dslr = "office-31//dslr//images//"
dataroot_webcam = "office-31//webcam//images//"

batch_size = 1
image_size = 64
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset_amazon = dset.ImageFolder(root=dataroot_amazon,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset_dslr = dset.ImageFolder(root=dataroot_dslr,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataset_webcam = dset.ImageFolder(root=dataroot_webcam,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader_amazon = torch.utils.data.DataLoader(dataset_amazon, batch_size=batch_size,
                                         shuffle=True)
dataloader_dslr = torch.utils.data.DataLoader(dataset_dslr, batch_size=batch_size,
                                         shuffle=True)
dataloader_webcam = torch.utils.data.DataLoader(dataset_webcam, batch_size=batch_size,
                                         shuffle=True)

print('finished loading the datasets.')

imgs_amazon = all_imgs_from_dataloader(dataloader_amazon, 'amazon')
imgs_dslr = all_imgs_from_dataloader(dataloader_dslr, 'dslr')
imgs_webcam = all_imgs_from_dataloader(dataloader_webcam, 'webcam')

################################################################################

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(img_dslr)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))