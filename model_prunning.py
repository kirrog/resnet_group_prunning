import copy
import gc
import json
import os
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from regularizations import filter_regularization_feature_from_weights, filter_regularization_feature_from_entropy, \
    filter_regularization_feature_from_rademacher, filter_regularization_feature_from_entropy_inv, \
    filter_regularization_feature_from_rademacher_inv
from src.batterfly_dataset_loader import BatterflyCSTMDatasetCreator
from src.blood_cells_dataset_loader import BloodCellsCSTMDatasetCreator
from src.brests_histopathology_dataset_loader import BreastHistCSTMDatasetCreator
from src.cifar100_dataset_loader import Cifar100CSTMDatasetCreator
from src.cifar10_dataset_loader import Cifar10CSTMDatasetCreator
from src.crop_disease_dataset_loader import CropDiseaseCSTMDatasetCreator
from src.flowers_dataset_loader import FlowersCSTMDatasetCreator
from src.fourniture_dataset_loader import FournitureCSTMDatasetCreator
from src.house_plant_dataset_loader import HousePlantCSTMDatasetCreator
from src.model import ResidualBlock, ResNet, rademacher_complexity, inner_data_entropy, inner_data_weights, \
    inner_data_entropy_inv, rademacher_complexity_inv, neg_inner_data_weights
from src.vehicles_dataset_loader import VehiclesCSTMDatasetCreator
from validation import validate_model

# ("cifar10", Cifar10CSTMDatasetCreator, 10)
# ("cifar100", Cifar100CSTMDatasetCreator, 100)
# ("butterfly", BatterflyCSTMDatasetCreator, 75)
# ("bloodcells", BloodCellsCSTMDatasetCreator, 8)
# ("crop", CropDiseaseCSTMDatasetCreator, 22)
# ("flowers", FlowersCSTMDatasetCreator, 5)
# ("fourniture", FournitureCSTMDatasetCreator, 32)
# ("houseplant", HousePlantCSTMDatasetCreator, 47)
# ("vehicle", VehiclesCSTMDatasetCreator, 7)
# ("tinyimagenet", TinyImagenetCSTMDatasetCreator, 200)
# ("breast", BreastHistCSTMDatasetCreator, 2)
# ("imagenet1k", Imagenet1kCSTMDatasetCreator, 1000)

dataset_name = "flowers"
dataset_num_classes = 5
dataset_creator_class = FlowersCSTMDatasetCreator


def init_model(mode_path: Path, num_classes):
    model = ResNet(ResidualBlock, [3, 1, 1, 3], num_classes)
    model.load_state_dict(torch.load(mode_path))
    model.eval()
    model = model.to("cuda")
    return model


def validate_and_calc_features_model(model, reg_func, test_loader, device, criterion):
    gc.collect()
    if reg_func is not None:
        for l in [model.layer0, model.layer1, model.layer2, model.layer3]:
            for seq in l:
                seq.is_processing = True
                seq.proc_func = reg_func
                del seq.inner_data
                gc.collect()
                seq.inner_data = []
                seq.inner_data_augmented = []
                seq.inner_data_used = True
                seq.aug_processing = False

    acc, loss = validate_model(model, list(test_loader[0]), device, criterion)
    if reg_func is not None:
        for l in [model.layer0, model.layer1, model.layer2, model.layer3]:
            for seq in l:
                seq.aug_processing = True
    acc_of_aug, loss_of_aug = validate_model(model, list(test_loader[1]), device, criterion)

    if reg_func is not None:
        for l in [model.layer0, model.layer1, model.layer2, model.layer3]:
            for seq in l:
                seq.is_processing = False
                seq.aug_processing = False
    return acc, loss


def make_step(model, test_loader, device, criterion, layer_num, num2delete, inner_data_reg_func, weights_reg_func):
    stats = dict()

    all_features0, lowest_feature_value0, size_value0 = model.recreation_with_filter_lowest_feature_delete(
        layer_num,
        num2delete, weights_reg_func, device)

    acc, loss = validate_and_calc_features_model(model, inner_data_reg_func, test_loader, device, criterion)

    stats["acc"] = acc
    stats["loss"] = float(loss.cpu())
    stats["all_features"] = all_features0
    stats["lowest_feature_value"] = lowest_feature_value0
    stats["size_value"] = size_value0

    return acc, stats


def search_by_prunning(criterion,
                       test_loader,
                       inner_data_regularization_function,
                       inner_data_regularization_name,
                       weights_regularization_function,
                       weights_regularization_name,
                       device,
                       init_step_sizes,
                       model_path_out: Path,
                       model_path_in: Path):
    print(model_path_in)
    model = init_model(model_path_in, dataset_num_classes)

    init_acc, init_loss = validate_and_calc_features_model(model, inner_data_regularization_function, test_loader,
                                                           device,
                                                           criterion)
    prev_model = copy.deepcopy(model).cpu()

    print(f"Inner: {inner_data_regularization_name}. "
          f"Weights: {weights_regularization_name}. "
          f"Init model. Test dataset: accuracy: {init_acc:0.5f} loss: {init_loss:0.5f}")

    current_step_sizes = init_step_sizes
    current_steps = [0, 0, 0, 0]
    max_steps = [64, 64, 512, 512]
    search_stats = dict()
    current_acc = init_acc
    step_acc = current_acc
    pbar = tqdm(range(256), total=256, desc="Search for best configuration")
    save_res = True
    with torch.no_grad():
        for i in pbar:
            current_stats = []
            for layer_num in range(4):
                if current_steps[layer_num] + 2 * current_step_sizes[layer_num] >= max_steps[layer_num]:
                    current_stats.append((layer_num, 0.0, -(current_acc - 0.0) / stats["size_value"]))
                    continue
                model = copy.deepcopy(prev_model).cuda()
                current_steps_changes = current_steps.copy()
                current_steps_changes[layer_num] += current_step_sizes[layer_num]
                pos_str = "_".join([str(x) for x in current_steps_changes])

                pbar.set_postfix_str(f"step: {layer_num}/4 "
                                     f"currant_acc: {current_acc:0.5f} "
                                     f"step_acc: {step_acc:0.5f} "
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
                                                inner_data_regularization_function, weights_regularization_function)
                    search_stats[pos_str] = (step_acc, stats)

                current_stats.append((layer_num, step_acc, -(current_acc - step_acc) / stats["size_value"]))
                del model
            layer2delete, new_acc, per_acc_drop_weight = list(sorted(current_stats, key=lambda x: x[2]))[-1]
            if new_acc + possible_drop < init_acc:
                print(f"Found best cut: acc: {current_acc:0.5f} 2cut: {current_steps}")
                break
            else:
                current_acc = new_acc
                current_steps[layer2delete] += current_step_sizes[layer2delete]
                model = copy.deepcopy(prev_model).cuda()
                all_features0, lowest_feature_value0, size_value0 = model.recreation_with_filter_lowest_feature_delete(
                    layer2delete,
                    current_step_sizes[layer2delete], weights_regularization_function, device)
                acc, loss = validate_and_calc_features_model(model, inner_data_regularization_function, test_loader,
                                                             device,
                                                             criterion)
                prev_model = copy.deepcopy(model).cpu()
                # if i > 3:
                #     save_res=False
                #     break

    if save_res:
        with open(model_path_out / f"init_{Path(model_path_in).name[:-4]}__"
                                   f"inner_reg_{inner_data_regularization_name}__"
                                   f"weights_reg_{weights_regularization_name}__"
                                   f"stats.json", "w", encoding="UTF-8") as f:
            json.dump(search_stats, f, ensure_ascii=False)

    del prev_model
    torch.cuda.empty_cache()
    gc.collect()


possible_drop = 0.0


# inner_regularization_functions = [
#     ("none", None),
#     ("weight", inner_data_weights),
#     ("new_weight", neg_inner_data_weights),
#     ("entr", inner_data_entropy),
#     ("entr-inv", inner_data_entropy_inv),
#     ("radem", rademacher_complexity),
#     ("radem-inv", rademacher_complexity_inv)
# ]


def ebm_stub(inner_data):
    dims2sum = list(range(len(inner_data.size())))[2:]  # get 2 last dums
    res = torch.sum(torch.abs(inner_data), dim=dims2sum)
    return res


inner_regularization_functions = [
    ("ebm", ebm_stub),
]

weights_regularization_functions = [
    ("none", None),
    ("weight", filter_regularization_feature_from_weights),
    # ("entr", filter_regularization_feature_from_entropy),
    # ("entr-inv", filter_regularization_feature_from_entropy_inv),
    # ("radem", filter_regularization_feature_from_rademacher),
    # ("radem-inv", filter_regularization_feature_from_rademacher_inv)
]

init_step_sizes = [1, 1, 8, 8]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_path = Path(f"./model_ebm/{dataset_name}_v1_pos_drop_{possible_drop}")
output_path.mkdir(parents=True, exist_ok=True)

dataset_creator = dataset_creator_class()

# test_loader = list(dataset_creator.create_loaders(create_test_dataloader=True)["test"])
test_loader_base = dataset_creator.create_loaders(create_test_dataloader=True)["test"]
test_loader_energy = dataset_creator.create_loaders(create_test_dataloader=True,
                                                    energy_test_transforms_bool=True)["test"]
test_loader = (test_loader_base, test_loader_energy)

criterion = nn.CrossEntropyLoss()
points_per_experiment = 3

experiments_ignore_list = [
    "/media/kirrog/Expansion/models/2025_04_11__01_43___flowers__v2__weights__1e-09_1e-08/ep_011_acc_0.736264.bin"
]
experiments_root_dir = Path("/media/kirrog/Expansion/models")
experiments_list = list(experiments_root_dir.glob(f"*{dataset_name}_*"))

print(f"Experiments amount: {len(experiments_list)}")
tasks = []
possible_tasks = []
num_epoch_list = []
for experiment_path in experiments_list:

    experiment_output_path = output_path / str(experiment_path.name)
    experiment_output_path.mkdir(parents=True, exist_ok=True)
    epoches_paths = list(experiment_path.glob('*.bin'))
    exp_features = str(experiment_path.name).split("__")
    reg_name = "none"
    if len(exp_features) > 3:
        reg_name = exp_features[4]
        if reg_name == "radem_v2-inv":
            reg_name = "radem-inv"
        if reg_name == "radem_v2":
            reg_name = "radem"
        if reg_name == "entropy":
            reg_name = "entr"
        if reg_name == "entropy-inv":
            reg_name = "entr-inv"
        if reg_name == "weights":
            reg_name = "weight"
    if reg_name not in ["none", "weight"]:
        continue
    epoches_best_paths = list(
        sorted([(x, float(str(x.name).split("_")[-1][:-4])) for x in epoches_paths], key=lambda x: x[1]))[
        -points_per_experiment:]
    num_epoch_list.append(len(epoches_best_paths))
    jsons_paths_exists_list = [str(x) for x in experiment_output_path.glob("*")]
    for epoch_path, acc in epoches_best_paths:
        if str(epoch_path) in experiments_ignore_list:
            print(f"IGNORE LIST: {epoch_path}")
            continue
        for inner_regularization_name, inner_regularization_function in inner_regularization_functions:
            for weights_regularization_name, weights_regularization_function in weights_regularization_functions:
                json_path_out = experiment_output_path / (f"init_{Path(epoch_path).name[:-4]}__"
                                                          f"inner_reg_{inner_regularization_name}__"
                                                          f"weights_reg_{weights_regularization_name}__"
                                                          f"stats.json")
                possible_tasks.append(json_path_out)
                if str(json_path_out) in jsons_paths_exists_list:
                    jsons_paths_exists_list.remove(str(json_path_out))
                if inner_regularization_name == "none" and weights_regularization_name == "none":
                    continue
                if weights_regularization_name != "none" and weights_regularization_name != reg_name:
                    continue
                if not json_path_out.exists():
                    tasks.append(
                        (inner_regularization_function, inner_regularization_name, weights_regularization_function,
                         weights_regularization_name, experiment_output_path, epoch_path))
    for json_rm_path in jsons_paths_exists_list:
        os.remove(json_rm_path)
        print(f"File removed: {json_rm_path}")
print(f"Tasks amount: {len(tasks)}")
print(f"Possible tasks amount: {len(possible_tasks)}")
print(f"Epoches amount: {sum(num_epoch_list)}")

for (inner_regularization_function,
     inner_regularization_name,
     weights_regularization_function,
     weights_regularization_name,
     experiment_output_path,
     epoch_path) in tasks:
    search_by_prunning(criterion,
                       test_loader,
                       inner_regularization_function,
                       inner_regularization_name,
                       weights_regularization_function,
                       weights_regularization_name,
                       device,
                       init_step_sizes,
                       experiment_output_path,
                       epoch_path)
