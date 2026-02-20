import torchvision.transforms as transforms
import torchvision
import torch
import os
from torchvision.datasets import ImageFolder

import numpy as np
import random

def get_loaders(dataset: str, batch_size: int, data_root: str = "./data", seed = 0):
    if dataset == "Cifar100":
        ds = torchvision.datasets.CIFAR100
        mean, std = (0.5071, 0.4867, 0.4409), (0.267, 0.256, 0.276)
        input_size = 32
    elif dataset == "Cifar10":
        ds = torchvision.datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        input_size = 32
    elif dataset == "SVHN":
        ds = torchvision.datasets.SVHN
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        input_size = 32
    elif dataset == "TinyImageNet":
        # Tiny‑ImageNet‑200 isn't bundled with torchvision, so we'll load it via
        # ImageFolder once it has been downloaded & extracted:
        #   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
        #   unzip tiny-imagenet-200.zip -d ./data
        # Directory structure after extraction:
        #   ./data/tiny-imagenet-200/train/*
        #   ./data/tiny-imagenet-200/val/images/*
        ds = None  # handled separately below
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        input_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    num_workers = 8 if os.name == "posix" else 0

    # ── transforms ────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size + 8),  # e.g. 72 → 64 center‑crop
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── dataset objects ───────────────────────────────────────────────────────
    if dataset in {"Cifar10", "Cifar100"}:
        trainset = ds(root=data_root, train=True, download=True, transform=transform_train)
        testset = ds(root=data_root, train=False, download=True, transform=transform_test)
    elif dataset == "SVHN":
        trainset = ds(root=data_root, split='train', download=True, transform=transform_train)
        testset = ds(root=data_root, split='test', download=True, transform=transform_test)
    else:  # Tiny‑ImageNet via ImageFolder
        train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")
        val_dir   = os.path.join(data_root, "tiny-imagenet-200", "val", "images_by_class")
        trainset = ImageFolder(train_dir, transform=transform_train)
        testset  = ImageFolder(val_dir,   transform=transform_test)

    # ── dataloaders ───────────────────────────────────────────────────────────

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return trainloader, testloader


class DataHelper:
    """Simple wrapper exposing `.trainloader`, `.testloader`, and metadata."""

    def __init__(self, dataset: str, batch_size: int, data_root: str = "./data", seed = 0):
        self.trainloader, self.testloader = get_loaders(dataset, batch_size, data_root, seed)
        self.name = dataset
        if dataset == "Cifar100":
            self.class_num = 100
        elif dataset == "Cifar10":
            self.class_num = 10
        elif dataset == "SVHN":
            self.class_num = 10
        elif dataset == "TinyImageNet":
            self.class_num = 200
        else:
            raise ValueError(f"Unknown dataset: {dataset}")



def Cifar10(batch_size: int = 128, data_root: str = "./data", seed = 0) -> DataHelper:
    return DataHelper("Cifar10", batch_size, data_root, seed)


def Cifar100(batch_size: int = 128, data_root: str = "./data", seed = 0) -> DataHelper:
    return DataHelper("Cifar100", batch_size, data_root, seed)


def SVHN(batch_size: int = 128, data_root: str = "./data", seed = 0) -> DataHelper:
    return DataHelper("SVHN", batch_size, data_root, seed)


def TinyImageNet(batch_size: int = 128, data_root: str = "./data", seed = 0) -> DataHelper:
    return DataHelper("TinyImageNet", batch_size, data_root, seed)
