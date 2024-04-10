import gc
from pathlib import Path

import torch
from torchsummary import summary
from tqdm import tqdm

from dataset_loader import data_loader
from src.model import ResidualBlock, ResNet

threshold = 0.02
experiment_num = 9
find = True
experiment_path = Path("/home/kirrog/projects/FQWB/model/4_block_cut")
experiment_path.mkdir(parents=True, exist_ok=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
MODEL_PATH_IN = "/home/kirrog/projects/FQWB/model/aug_4_block_reg_group/l1_0.0001_l2_1e-05_wd_1e-08/result_acc_0.807900.bin"
model = ResNet(ResidualBlock, [3, 1, 1, 3])
model.load_state_dict(torch.load(MODEL_PATH_IN))
model.eval()
# model = model.cuda()
summary(model, (3, 32, 32))
# exit(0)
batch_size = 100
test_loader = data_loader(data_dir='./data',
                          batch_size=batch_size,
                          test=True)
model = model.cpu()
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
model = model.cpu()
model.recreation_with_filter_inner_data_regularization(threshold, test_loader)
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
