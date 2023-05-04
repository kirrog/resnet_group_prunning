import gc
from pathlib import Path

import torch

from dataset_loader import data_loader
from src.model import ResidualBlock, ResNet

threshold = 0.00145
experiment_num = 9
find = True
experiment_path = Path("/home/kirrog/projects/FQWB/model/unique_feature_threshold")
experiment_path.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH_IN = "/home/kirrog/projects/FQWB/model/unique_feature/l1_1e-09_l2_1e-10_wd_1e-08/ep_029_acc_0.814800.bin"
model = ResNet(ResidualBlock, [3, 4, 6, 3])
model.load_state_dict(torch.load(MODEL_PATH_IN))
model.eval()
model = model.cuda()
# summary(model, (3, 224, 224))
batch_size = 100
test_loader = data_loader(data_dir='./data',
                          batch_size=batch_size,
                          test=True)
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
model.eval()
model = model.cuda()
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
