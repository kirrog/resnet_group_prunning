import gc
from pathlib import Path

import torch.nn as nn
from tqdm import tqdm

from dataset_loader import data_loader
from regularizations import *
from src.model import ResNet, ResidualBlock
from validation import validate_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256
# CIFAR10 dataset
train_loader, valid_loader = data_loader(data_dir='./data',
                                         batch_size=batch_size)
experiment_name = "aug_5_blocks_and_maxpool"
test_loader = data_loader(data_dir='./data',
                          batch_size=batch_size,
                          test=True)
# iter_range = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
iter_range = [0, 0]
num_classes = 10
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-8

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
    experiment_path = Path(f"/home/kirrog/projects/FQWB/model/{experiment_name}/{experiment}")
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

            # for params in elems:
            #     weights, bias, norm_coef, norm_bias = params
            #     loss += regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, weight_coef_l1,
            #                                              weight_coef_l2)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
        ep = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} ' \
             f'Weight value: ' \
             f'{calc_mean_weights(model):0.4f}'
        print(ep)

        # Validation
        acc = validate_model(model, valid_loader, device)
        ac = 'Accuracy of the network on the {} validation images: {} %'.format(len(valid_loader), 100 * acc)
        print(ac)
        with open(str(experiment_path / "stats.txt"), "a") as f:
            f.write(f"{ep}\n")
            f.write(f"{ac}\n")
        torch.save(model.state_dict(), str(experiment_path / f"ep_{epoch:03d}_acc_{acc:04f}.bin"))

    acc = validate_model(model, test_loader, device)
    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * acc))
    torch.save(model.state_dict(), str(experiment_path / f"result_acc_{acc:04f}.bin"))
    del model
    torch.cuda.empty_cache()
    gc.collect()
