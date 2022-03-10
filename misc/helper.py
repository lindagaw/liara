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
from PIL import Image

from torch.utils.data import Dataset

import torchvision
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def all_imgs_from_dataloader(dataloder, name):
    images = []
    for i, (inputs, labels) in enumerate(dataloder):
        images.append(np.squeeze(inputs.numpy()))

    images = np.asarray(images)
    print('the shape of all the images in dataloader {} is {}'.format(name, images.shape) )

    return images

def save_individual_images(path_to_save_to, tensor):
    try:
        os.makedirs(path_to_save_to)
    except:
        pass

    for i in range(0, len(tensor)):
        data = tensor[i]
        path = path_to_save_to + str(i) + '.png'
        torchvision.utils.save_image(data, path)

import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net

def get_particular_class(dataset, category, order):
    try:
        targets = dataset.targets
    except:
        targets = dataset.labels

    data = dataset.data
    new_targets = []
    new_data = []

    for target, sample in zip(targets, data):
        if target == category:
            new_targets.append(target.numpy())
            if order == 'svhn':
                new_data.append(sample.transpose(2,1,0))
            else:
                new_data.append(sample.numpy())

    return new_data, new_targets


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def remove_non_pic(folder):

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith('.csv'):
            os.remove(path)
            print('removed '.format(path))
