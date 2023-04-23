from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import gc
from proxssi.groups import resnet_groups
from src.model import ResNet, ResidualBlock

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.jit.script
def l1_l2_loss(param, wcl1, wcl2):
    res = torch.sum(torch.abs(param)) * wcl1
    res += torch.sum(param ** 2) * wcl2
    return res


@torch.jit.script
def l1_l2_loss_biased(param, wcl1, wcl2):
    param_biased = param - 1.0
    res = torch.sum(torch.abs(param_biased)) * wcl1
    res += torch.sum(param_biased ** 2) * wcl2
    return res


@torch.jit.script
def regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, wcl1, wcl2):
    res = torch.zeros((1)).cuda()
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2)
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


batch_size = 208
# CIFAR10 dataset
train_loader, valid_loader = data_loader(data_dir='./data',
                                         batch_size=batch_size)

test_loader = data_loader(data_dir='./data',
                          batch_size=batch_size,
                          test=True)
iter_range = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
num_classes = 10
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-8


class Arguments:
    learning_rate: float = learning_rate
    weight_decay: float = weight_decay


for i in range(len(iter_range) - 1):
    weight_coef_l1 = torch.as_tensor(iter_range[i + 1]).to(device)
    weight_coef_l2 = torch.as_tensor(iter_range[i]).to(device)
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(resnet_groups_optimizer, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    # Train the model
    total_step = len(train_loader)

    experiment = f"l1_{iter_range[i + 1]}_l2_{iter_range[i]}_wd_{weight_decay}"
    experiment_path = Path(f"/home/kirrog/projects/FQWB/model/unique_feature/{experiment}")
    experiment_path.mkdir(exist_ok=True, parents=True)
    l = 0
    elems = []
    last_elem = []
    for param in model.parameters():
        if l % 4 == 0:
            last_elem = []
        last_elem.append(param)
        if l % 4 == 3:
            elems.append(last_elem)
        l += 1

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader, desc="training")):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            for params in elems:
                weights, bias, norm_coef, norm_bias = params
                loss += regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, weight_coef_l1,
                                                         weight_coef_l2)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
        ep = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} Weight value: {sum([float(torch.sum(x)) for x in model.parameters()]):0.4f}'
        print(ep)

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            acc = correct / total
            ac = 'Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * acc)
            print(ac)
        with open(str(experiment_path / "stats.txt"), "a") as f:
            f.write(f"{ep}\n")
            f.write(f"{ac}\n")
        torch.save(model.state_dict(), str(experiment_path / f"ep_{epoch:03d}_acc_{acc:04f}.bin"))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        acc = correct / total
        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * acc))
        torch.save(model.state_dict(), str(experiment_path / f"result_acc_{acc:04f}.bin"))
    del model
    torch.cuda.empty_cache()
    gc.collect()
