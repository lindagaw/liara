"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

import params

def get_stl_10_datasets():
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
    stl_10_dataset_pseudo_train = datasets.ImageFolder(root="datasets//stl_10_pseudo_train",
                               transform=transform)
    stl_10_dataset_train = datasets.ImageFolder(root="datasets//stl_10_train",
                               transform=transform)
    stl_10_dataset_test = datasets.ImageFolder(root="datasets//stl_10_test",
                               transform=transform)

    return stl_10_dataset_pseudo_train, stl_10_dataset_train, stl_10_dataset_test


def get_stl_10():
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
    stl_10_dataset_pseudo_train = datasets.ImageFolder(root="datasets//stl_10_pseudo_train",
                               transform=transform)
    stl_10_dataset_train = datasets.ImageFolder(root="datasets//stl_10_train",
                               transform=transform)
    stl_10_dataset_test = datasets.ImageFolder(root="datasets//stl_10_test",
                               transform=transform)

    stl_10_data_loader_pseudo_train = torch.utils.data.DataLoader(
        dataset=stl_10_dataset_pseudo_train,
        batch_size=params.batch_size,
        shuffle=True)

    stl_10_data_loader_train = torch.utils.data.DataLoader(
        dataset=stl_10_dataset_train,
        batch_size=params.batch_size,
        shuffle=True)

    stl_10_data_loader_test = torch.utils.data.DataLoader(
        dataset=stl_10_test,
        batch_size=params.batch_size,
        shuffle=True)

    return stl_10_data_loader_pseudo_train, stl_10_data_loader_train, stl_10_data_loader_test
