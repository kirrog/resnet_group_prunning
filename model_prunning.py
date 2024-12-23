import copy
import gc
import json
from pathlib import Path

import torch
from torch import nn
from torchsummary import summary
from tqdm import tqdm

from src.dataset_loader import data_loader, Cifar10CSTMDatasetCreator
from src.model import ResidualBlock, ResNet, rademacher_complexity
from validation import validate_model


def init_model(mode_path: Path):
    model = ResNet(ResidualBlock, [3, 1, 1, 3])
    model.load_state_dict(torch.load(mode_path))
    model.eval()
    model = model.to("cuda")
    return model


def validate_and_calc_features_model(model, reg_func, test_loader, device, criterion):
    gc.collect()
    for l in [model.layer0, model.layer1, model.layer2, model.layer3]:
        for seq in l:
            seq.is_processing = True
            seq.proc_func = reg_func
            del seq.inner_data
            gc.collect()
            seq.inner_data = []
    acc, loss = validate_model(model, test_loader, device, criterion)
    for l in [model.layer0, model.layer1, model.layer2, model.layer3]:
        for seq in l:
            seq.is_processing = False
    return acc, loss


def make_step(model, test_loader, device, criterion, layer_num, num2delete, reg_func):
    stats = dict()

    all_features0, lowest_feature_value0, size_value0 = model.recreation_with_filter_lowest_delete(
        layer_num,
        num2delete)

    acc, loss = validate_and_calc_features_model(model, reg_func, test_loader, device, criterion)

    stats["acc"] = acc
    stats["loss"] = loss
    stats["all_features"] = all_features0
    stats["lowest_feature_value"] = lowest_feature_value0
    stats["size_value"] = size_value0

    return acc, stats


possible_drop = 0.002
regularization_function = rademacher_complexity

experiment_path = Path("/home/kirrog/projects/FQWB/model/v1")
experiment_path.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_in = Path("/media/kirrog/data/data/fqwb_data/models/2024_12_07__13_34___cifar10/ep_047_acc_0.846400.bin")
model = init_model(model_path_in)
prev_model = copy.deepcopy(model).cpu()

# batch_size = 100

cifar10_dataset_creator = Cifar10CSTMDatasetCreator()
test_loader = cifar10_dataset_creator.create_loaders(create_test_dataloader=True)["test"]
criterion = nn.CrossEntropyLoss()

init_acc, init_loss = validate_and_calc_features_model(model, regularization_function, test_loader, device, criterion)

print(f"Init model. Test dataset: accuracy: {init_acc} loss: {init_loss}")

num2delete = 1
current_step_sizes = [1, 2, 4, 8]
current_steps = [0, 0, 0, 0]
search_stats = dict()
current_acc = init_acc
step_acc = current_acc
pbar = tqdm(range(256), total=256, desc="Search for best configuration")
with torch.no_grad():
    for i in pbar:
        current_stats = []

        for layer_num in range(4):
            model = copy.deepcopy(prev_model).cuda()
            current_steps_changes = current_steps.copy()
            current_steps_changes[layer_num] += current_step_sizes[layer_num]
            pos_str = "_".join([str(x) for x in current_steps_changes])

            pbar.set_postfix_str(f"step: {layer_num}/4 "
                                 f"currant_acc: {current_acc} "
                                 f"step_acc: {step_acc} "
                                 f"current_steps: {pos_str}")
            if pos_str in search_stats:
                step_acc, stats = search_stats[pos_str]
            else:
                step_acc, stats = make_step(model,
                                            test_loader,
                                            device,
                                            criterion,
                                            layer_num,
                                            current_step_sizes[layer_num],
                                            regularization_function)
                search_stats[pos_str] = (step_acc, stats)

            current_stats.append((layer_num, step_acc))
            del model
        layer2delete, new_acc = list(sorted(current_stats, key=lambda x: x[1]))[-1]
        if new_acc + possible_drop < init_acc:
            print(f"Found best cut: acc: {current_acc} 2cut: {current_steps}")
            break
        else:
            current_acc = new_acc
            current_steps[layer2delete] += current_step_sizes[layer2delete]
            model = copy.deepcopy(prev_model).cuda()
            all_features0, lowest_feature_value0, size_value0 = model.recreation_with_filter_lowest_delete(
                layer2delete,
                current_step_sizes[layer2delete])
            acc, loss = validate_and_calc_features_model(model, regularization_function, test_loader, device, criterion)
            prev_model = copy.deepcopy(model).cpu()

with open(experiment_path / f"init_{Path(model_path_in).name[:-4]}_stats.json", "w", encoding="UTF-8") as f:
    json.dump(search_stats, f, ensure_ascii=False)

acc, loss = validate_model(model, test_loader, device, criterion)
print(f"Cut model. Test dataset: accuracy: {acc} loss: {loss}")
torch.save(prev_model.state_dict(),
           str(experiment_path / f"init_{Path(model_path_in).name[:-4]}_"
                                 f"steps_{current_steps}_"
                                 f"result_acc_{acc:04f}.bin"))
del prev_model
torch.cuda.empty_cache()
gc.collect()
