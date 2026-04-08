import gc
from pathlib import Path

import torch
from torchsummary import summary
from tqdm import tqdm

from src.cifar10_dataset_loader import Cifar10CSTMDatasetCreator
from src.model import ResidualBlock, ResNet, rademacher_complexity

threshold = 32.0
experiment_num = 9
find = True
experiment_path = Path("/home/kirrog/projects/FQWB/model/v1")
experiment_path.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
MODEL_PATH_IN = "/media/kirrog/data/data/fqwb_data/models/2024_12_07__13_34___cifar10/ep_047_acc_0.846400.bin"
model = ResNet(ResidualBlock, [3, 1, 1, 3])
model.load_state_dict(torch.load(MODEL_PATH_IN))
model.eval()
model = model.to("cuda")
# model = model.cuda()
summary(model, (3, 32, 32))
# exit(0)
batch_size = 100
cifar10_dataset_creator = Cifar10CSTMDatasetCreator(batch_size=batch_size)
test_loader = cifar10_dataset_creator.create_loaders(create_test_dataloader=True)["test"]
model = model.to(device)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs
    acc = correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * acc))
model = model.to(device)
model.recreation_with_filter_inner_data_regularization_by_func(threshold, test_loader, rademacher_complexity)
model.eval()
# model = model.cuda()
# summary(model, (3, 224, 224))
# if find:
#     exit(0)
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
    print('Accuracy of the cutted network on the {} test images: {} %'.format(10000, 100 * acc))
    torch.save(model.state_dict(),
               str(experiment_path / f"{experiment_num}_{Path(MODEL_PATH_IN).name[:-4]}_threshold_{threshold}_result_acc_{acc:04f}.bin"))
del model
torch.cuda.empty_cache()
gc.collect()
