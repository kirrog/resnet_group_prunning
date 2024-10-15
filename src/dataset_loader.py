import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from src.loggers import create_logger

logger = create_logger(__name__)


class Cifar10CSTMDatasetCreator:
    def __init__(self, data_dir: Path = Path("/media/kirrog/data/data/fqwb_data/data/cifar_10"),
                 batch_size: int = 1024,
                 random_seed: int = 42,
                 valid_size: float = 0.1,
                 shuffle: bool = True,
                 num_of_workers: int = 24,
                 image_size: Tuple[int, int] = (32, 32)):
        if not (data_dir.exists() and data_dir.is_dir()):
            logger.warning(f"Datadir looks wrong: "
                           f"exists: {data_dir.exists()} "
                           f"is_dir: {data_dir.is_dir()} "
                           f"text: {data_dir}")
        else:
            logger.info("Datapath is existed dir")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.num_of_workers = num_of_workers
        self.image_size = image_size
        logger.info("Cifar10DataCreator init")

    def create_loaders(self, create_test_dataloader: bool = False) -> Dict[str, torch.utils.data.DataLoader]:
        logger.info(f"Create logger: is_test:{create_test_dataloader}")
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        if create_test_dataloader:
            dataset = datasets.CIFAR10(
                root=self.data_dir, train=False,
                download=True, transform=transform,
            )

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_of_workers  # , pin_memory=True
            )

            return {"test": data_loader}

        # load the dataset
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_of_workers)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_of_workers)

        return {"train": train_loader, "valid": valid_loader}


def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False,
                num_of_workers=24):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_of_workers  # , pin_memory=True
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_of_workers)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_of_workers)

    return (train_loader, valid_loader)


if __name__ == "__main__":
    logger = create_logger(__name__, True, False)
    logger.setLevel(logging.DEBUG)
    cifar10_dataset_creator = Cifar10CSTMDatasetCreator()
    test_dataloader = cifar10_dataset_creator.create_loaders(create_test_dataloader=True)["test"]
    train_valid_dataloaders = cifar10_dataset_creator.create_loaders()
    train_dataloader = train_valid_dataloaders["train"]
    valid_dataloader = train_valid_dataloaders["valid"]
