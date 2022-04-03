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

def balance(real, fake):
    # calculate np.linalg.norm for real for real ones
    real_norms = []
    for sample in real:
        real_norms.append(np.linalg.norm(sample))

    fake_norms = []
    for sample in fake:
        fake_norms.append(np.linalg.norm(sample))

    real_norms = torch.FloatTensor(real_norms)
    fake_norms = torch.FloatTensor(fake_norms)

    loss = nn.MSELoss(real_norms, fake_norms)

    return loss
