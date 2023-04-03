import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from src.model import ResidualBlock, ResNet


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


threshold = 2.0325
experiment_num = 9
find = False
experiment_path = Path("/home/kirrog/projects/FQWB/model/threshold")
experiment_path.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH_IN = "/home/kirrog/projects/FQWB/model/l1_1e-09_l2_1e-10_wd_1e-08/ep_025_acc_0.822600.bin"
model = ResNet(ResidualBlock, [3, 4, 6, 3])
model.load_state_dict(torch.load(MODEL_PATH_IN))
model.eval()
summary(model, (3, 224, 224))
batch_size = 100
test_loader = data_loader(data_dir='./data',
                          batch_size=batch_size,
                          test=True)
if not find:
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
model.recreation(threshold)
if find:
    exit(0)
summary(model, (3, 224, 224))
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
    torch.save(model.state_dict(),
               str(experiment_path / f"{experiment_num}_{Path(MODEL_PATH_IN).name[:-4]}_threshold_{threshold}_result_acc_{acc:04f}.bin"))
del model
torch.cuda.empty_cache()
gc.collect()
