#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate mean and std."""

__author__ = 'Chong Guo <armourcy@gmail.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'MIT'

import os
import numpy as np

from torchvision import datasets, transforms

train_transform = transforms.Compose([transforms.ToTensor()])

# cifar10
train_set = datasets.CIFAR10(root='../data/', train=True, download=True, transform=train_transform)
print(train_set.data.shape)
print(train_set.data.mean(axis=(0,1,2))/255)
print(train_set.data.std(axis=(0,1,2))/255)
# (50000, 32, 32, 3)
# [0.49139968  0.48215841  0.44653091]
# [0.24703223  0.24348513  0.26158784]


# mnist
train_set = datasets.STL10(root='../data/', split='train', download=True, transform=train_transform)
print(train_set.data.shape)
print(train_set.data.mean(axis=(0,1,2))/255)
print(train_set.data.std(axis=(0,1,2))/255)
# [60000, 28, 28]
# 0.1306604762738429
# 0.30810780717887876
