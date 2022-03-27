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
import PIL
from PIL import Image

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

from misc import weights_init, save_individual_images, get_particular_class, get_same_index
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

category = 0

# Root directory for dataset
dataroot_fake = "generated_images//cifar_to_stl//" + str(category)

transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# Batch size during training

dataset_src_train = datasets.CIFAR10(root='./data',
                              train='train',
                              transform=transform,
                              download=True)
dataset_src_train.targets = torch.tensor(dataset_src_train.targets)
idx = get_same_index(dataset_src_train.targets, category)
dataset_src_train.targets= dataset_src_train.targets[idx]
dataset_src_train.data = dataset_src_train.data[idx]

dataset_tgt_train = datasets.STL10(root='./data',
                              split='train',
                              transform=transform,
                              download=True)
dataset_tgt_train.labels[dataset_tgt_train.labels == 1] = 99
dataset_tgt_train.labels[dataset_tgt_train.labels == 2] = 1
dataset_tgt_train.labels[dataset_tgt_train.labels == 99] = 2
dataset_tgt_train.labels[dataset_tgt_train.labels == 7] = 99
dataset_tgt_train.labels[dataset_tgt_train.labels == 6] = 7
dataset_tgt_train.labels[dataset_tgt_train.labels == 99] = 6

dataset_tgt_train.labels = torch.tensor(dataset_tgt_train.labels)
idx = get_same_index(dataset_tgt_train.labels, category)
dataset_tgt_train.labels = dataset_tgt_train.labels[idx]
dataset_tgt_train.data = dataset_tgt_train.data[idx]

fake_data = []
for file in os.listdir(dataroot_fake):
    image = np.asarray(Image.open(os.path.join(dataroot_fake, file)))
    fake_data.append(image)

fake_data = np.asarray(fake_data)


################################################################################
from skimage.transform import resize

def resize_images(images_array):
    imgs = []
    for image in images_array:
        image = resize(image, (image_size, image_size))
        a, b, c = image.shape
        image = image.reshape(a*b*c)
        imgs.append(image)
    return np.asarray(imgs)


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time


src_train_data = resize_images(dataset_src_train.data[:500])
red = src_train_data_y = [0]*len(src_train_data)

tgt_train_data = dataset_tgt_train.data
a, b, c, d = tgt_train_data.shape
tgt_train_data = resize_images(tgt_train_data.reshape(a, c, d, b))

fake_data = resize_images(fake_data)
green = tgt_train_data_y = [1]*len(tgt_train_data)
yellow = fake_data_y = [2]*len(fake_data)

print(src_train_data.shape)
print(tgt_train_data.shape)
print(fake_data.shape)

X = np.vstack((src_train_data, tgt_train_data, fake_data))
y = np.asarray(red + green + yellow)

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# We want to get TSNE embedding with 2 dimensions
n_components = 2
tsne = TSNE(n_components)
Y = tsne.fit_transform(X)
print(Y.shape)
# (1000, 2)
# Two dimensions for each of our images

# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE


red = get_same_index(y, 0)
green = get_same_index(y, 1)
yellow = get_same_index(y, 2)

fig, ax = plt.subplots(1)
ax.scatter(Y[red, 0], Y[red, 1], c="r")
ax.scatter(Y[green, 0], Y[green, 1], c="g")
ax.scatter(Y[yellow, 0], Y[yellow, 1], c="y")

ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis("tight")

plt.savefig('generated_images//tsne_cifar_fake_stl.png')
