import pickle
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import params

import os
import gzip
from torchvision import datasets, transforms

def get_office_home_datasets(domain):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize((0.5,), (0.5,))])
    # dataset and data loader
    office_home_dataset_pseudo_train = datasets.ImageFolder(root="datasets//" + domain + "office_home_pseudo_train",
                               transform=transform)
    office_home_dataset_train = datasets.ImageFolder(root="datasets//" + domain + "office_home_train",
                               transform=transform)
    office_home_dataset_test = datasets.ImageFolder(root="datasets//" + domain + "office_home_test",
                               transform=transform)

    return office_home_dataset_pseudo_train, office_home_dataset_train, office_home_dataset_test


def get_office_home(domain):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize((0.5,), (0.5,))])
    # dataset and data loader
    office_home_dataset_pseudo_train = datasets.ImageFolder(root="datasets//" + domain + "office_home_pseudo_train",
                               transform=transform)
    office_home_dataset_train = datasets.ImageFolder(root="datasets//" + domain + "office_home_train",
                               transform=transform)
    office_home_dataset_test = datasets.ImageFolder(root="datasets//" + domain + "office_home_test",
                               transform=transform)

    office_home_data_loader_pseudo_train = torch.utils.data.DataLoader(
        dataset=office_home_dataset_pseudo_train,
        batch_size=params.batch_size,
        shuffle=True)

    office_home_data_loader_train = torch.utils.data.DataLoader(
        dataset=office_home_dataset_train,
        batch_size=params.batch_size,
        shuffle=True)

    office_home_data_loader_test = torch.utils.data.DataLoader(
        dataset=office_home_test,
        batch_size=params.batch_size,
        shuffle=True)

    return office_home_data_loader_pseudo_train, office_home_data_loader_train, office_home_data_loader_test
