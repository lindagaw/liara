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

from datasets_code import get_stl_10, get_cifar_10, get_office_31, get_office_home
from datasets_code import get_stl_10_datasets, get_cifar_10_datasets, get_office_31_datasets, get_office_home_datasets
import pretty_errors
# Set random seed for reproducibility
manualSeed =999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 32
image_size = 299


dataset_amazon_webcam = datasets.ImageFolder(root='generated_images//office_31_amazon_to_webcam',
                           transform=transform)
dataset_amazon_dslr = datasets.ImageFolder(root='generated_images//office_31_dslr_to_webcam',
                           transform=transform)
dataset_dslr_webcam = datasets.ImageFolder(root='generated_images//office_31_dslr_to_webcam',
                           transform=transform)
dataset_webcam_amazon = datasets.ImageFolder(root='generated_images//office_31_webcam_amazon',
                           transform=transform)
dataset_dslr_amazon = datasets.ImageFolder(root='generated_images//office_31_dslr_amazon',
                           transform=transform)
dataset_webcam_dslr = datasets.ImageFolder(root='generated_images//office_31_webcam_to_dslr',
                           transform=transform)

dataset_pseudo_amazon, train_set_amazon, test_set_amazon = get_office_31_datasets('Amazon')
dataset_pseudo_dslr, train_set_dslr, test_set_dslr = get_office_31_datasets('Dslr')
dataset_pseudo_webcam, train_set_webcam, test_set_webcam = get_office_31_datasets('Webcam')

train_amazon_webcam = ConcatDataset((train_set_amazon, dataset_pseudo_webcam, dataset_amazon_webcam))
train_amazon_dslr = ConcatDataset((train_set_amazon, dataset_pseudo_dslr, dataset_amazon_dslr))
train_webcam_dslr = ConcatDataset((train_set_webcam, dataset_pseudo_dslr, dataset_webcam_dslr))
train_webcam_amazon = ConcatDataset((train_set_amazon, dataset_pseudo_webcam, dataset_webcam_amazon))
train_dslr_amazon = ConcatDataset((train_set_amazon, dataset_pseudo_dslr, dataset_dslr_amazon))
train_dslr_webcam = ConcatDataset((train_set_webcam, dataset_pseudo_dslr, dataset_dslr_webcam))
#train_amazon_dslr = ConcatDataset((train_set_amazon,dataset_amazon_dslr, train_set_dslr))
#train_amazon_webcam = ConcatDataset((train_set_amazon, dataset_amazon_webcam, train_set_webcam))
#train_webcam_dslr = ConcatDataset((train_set_webcam, dataset_dslr_webcam, train_set_dslr))

dataloader_train_amazon = torch.utils.data.DataLoader(train_set_amazon, batch_size=batch_size, shuffle=True)
dataloader_train_webcam = torch.utils.data.DataLoader(train_set_webcam, batch_size=batch_size, shuffle=True)
dataloader_train_dslr = torch.utils.data.DataLoader(train_set_dslr, batch_size=batch_size, shuffle=True)

dataloader_train_amazon_dslr = torch.utils.data.DataLoader(train_amazon_dslr, batch_size=batch_size, shuffle=True)
dataloader_train_amazon_webcam = torch.utils.data.DataLoader(train_amazon_webcam, batch_size=batch_size, shuffle=True)
dataloader_train_webcam_dslr = torch.utils.data.DataLoader(train_webcam_dslr, batch_size=batch_size, shuffle=True)
dataloader_train_dslr_amazon = torch.utils.data.DataLoader(train_dslr_amazon, batch_size=batch_size, shuffle=True)
dataloader_train_webcam_amazon = torch.utils.data.DataLoader(train_webcam_amazon, batch_size=batch_size, shuffle=True)
dataloader_train_dslr_webcam = torch.utils.data.DataLoader(train_dslr_webcam, batch_size=batch_size, shuffle=True)

dataloader_train_amazon = torch.utils.data.DataLoader(train_set_amazon, batch_size=batch_size, shuffle=True)
dataloader_train_webcam = torch.utils.data.DataLoader(train_set_webcam, batch_size=batch_size, shuffle=True)
dataloader_train_dslr = torch.utils.data.DataLoader(train_set_dslr, batch_size=batch_size, shuffle=True)

#dataloader_train_amazon = torch.utils.data.DataLoader(train_set_amazon, batch_size=batch_size, shuffle=True)
dataloader_test_amazon = torch.utils.data.DataLoader(test_set_amazon, batch_size=batch_size, shuffle=True)
#dataloader_train_dslr = torch.utils.data.DataLoader(train_set_dslr, batch_size=batch_size, shuffle=True)
dataloader_test_dslr = torch.utils.data.DataLoader(test_set_dslr, batch_size=batch_size, shuffle=True)
#dataloader_train_webcam = torch.utils.data.DataLoader(train_set_webcam, batch_size=batch_size, shuffle=True)
dataloader_test_webcam = torch.utils.data.DataLoader(test_set_webcam, batch_size=batch_size, shuffle=True)


f = get_classifier('inception_v3', pretrain=True)

print(f)

f.fc = nn.Linear(2048, 31)

classifier = f.cuda()

#classifier = train(classifier, dataloader_train_amazon_webcam)
#classifier = train(classifier, dataloader_train_amazon_dslr)
classifier = train(classifier, dataloader_train_webcam_dslr)

#classifier = train(classifier, dataloader_train_amazon)


print('eval on amazon')
acc = eval(classifier, dataloader_test_amazon)
print('eval on webcam')
acc = eval(classifier, dataloader_test_webcam)
print('eval on dslr')
acc = eval(classifier, dataloader_test_dslr)
