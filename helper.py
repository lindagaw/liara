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

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def all_imgs_from_dataloader(dataloder, name):
    images = None
    for i, (inputs, labels) in enumerate(dataloder):
        try:
            if images == None:
                images = inputs.numpy()
            else:
                images = np.vstack(images,inputs.numpy())
        except:
            continue

        #images.append(inputs.numpy())

    images = np.asarray(images[:-2])
    print('the shape of all the images in dataloader {} is {}'.format(name, images.shape) )

    return images
