"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

import params

def get_cifar_10_datasets():
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
    cifar_10_dataset_pseudo_train = datasets.ImageFolder(root="datasets//cifar_10_pseudo_train",
                               transform=transform)
    cifar_10_dataset_train = datasets.ImageFolder(root="datasets//cifar_10_train",
                               transform=transform)
    cifar_10_dataset_test = datasets.ImageFolder(root="datasets//cifar_10_test",
                               transform=transform)

    cifar_10_dataset_pseudo_train.targets[cifar_10_dataset_pseudo_train.targets == 2] = 100
    cifar_10_dataset_pseudo_train.targets[cifar_10_dataset_pseudo_train.targets == 1] = 2
    cifar_10_dataset_pseudo_train.targets[cifar_10_dataset_pseudo_train.targets == 100] = 1

    cifar_10_dataset_train.targets[cifar_10_dataset_train.targets == 2] = 100
    cifar_10_dataset_train.targets[cifar_10_dataset_train.targets == 1] = 2
    cifar_10_dataset_train.targets[cifar_10_dataset_train.targets == 100] = 1

    cifar_10_dataset_test.targets[cifar_10_dataset_test.targets == 2] = 100
    cifar_10_dataset_test.targets[cifar_10_dataset_test.targets == 1] = 2
    cifar_10_dataset_test.targets[cifar_10_dataset_test.targets == 100] = 1

    return cifar_10_dataset_pseudo_train, cifar_10_dataset_train, cifar_10_dataset_test


def get_cifar_10():
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
    cifar_10_dataset_pseudo_train = datasets.ImageFolder(root="datasets//cifar_10_pseudo_train",
                               transform=transform)
    cifar_10_dataset_train = datasets.ImageFolder(root="datasets//cifar_10_train",
                               transform=transform)
    cifar_10_dataset_test = datasets.ImageFolder(root="datasets//cifar_10_test",
                               transform=transform)

    cifar_10_dataset_pseudo_train.targets[cifar_10_dataset_pseudo_train.targets == 2] = 100
    cifar_10_dataset_pseudo_train.targets[cifar_10_dataset_pseudo_train.targets == 1] = 2
    cifar_10_dataset_pseudo_train.targets[cifar_10_dataset_pseudo_train.targets == 100] = 1

    cifar_10_dataset_train.targets[cifar_10_dataset_train.targets == 2] = 100
    cifar_10_dataset_train.targets[cifar_10_dataset_train.targets == 1] = 2
    cifar_10_dataset_train.targets[cifar_10_dataset_train.targets == 100] = 1

    cifar_10_dataset_test.targets[cifar_10_dataset_test.targets == 2] = 100
    cifar_10_dataset_test.targets[cifar_10_dataset_test.targets == 1] = 2
    cifar_10_dataset_test.targets[cifar_10_dataset_test.targets == 100] = 1

    cifar_10_data_loader_pseudo_train = torch.utils.data.DataLoader(
        dataset=cifar_10_dataset_pseudo_train,
        batch_size=params.batch_size,
        shuffle=True)

    cifar_10_data_loader_train = torch.utils.data.DataLoader(
        dataset=cifar_10_dataset_train,
        batch_size=params.batch_size,
        shuffle=True)

    cifar_10_data_loader_test = torch.utils.data.DataLoader(
        dataset=cifar_10_test,
        batch_size=params.batch_size,
        shuffle=True)

    return cifar_10_data_loader_pseudo_train, cifar_10_data_loader_train, cifar_10_data_loader_test
