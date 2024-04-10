import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from dataset_loader import data_loader
from metrics import calc_metrics
from src.model import ResNet, ResidualBlock

aug_4_block_path = Path("/home/kirrog/projects/FQWB/model/aug_4_block")
aug_4_block_reg_block_path = Path("/home/kirrog/projects/FQWB/model/aug_4_block_reg_block")
aug_4_block_reg_group_path = Path("/home/kirrog/projects/FQWB/model/aug_4_block_reg_group")
output_path = Path("/home/kirrog/projects/FQWB/model/stats_radamcher_entropy")

hyperparams_list = [aug_4_block_path, aug_4_block_reg_group_path, aug_4_block_reg_block_path]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_of_filter_steps = 10
num_of_block_steps = 10
num_of_filter_substeps = 10
num_of_block_substeps = 10
acceptable_loss_acc_value = 0.02


def get_new_model_instance():
    return ResNet(ResidualBlock, [3, 1, 1, 3])


def experiment_on_model_with_lowest_filter_entropy(model_path: Path, test_loader, num_of_layers=4):
    model = get_new_model_instance()
    orig_state = torch.load(str(model_path))
    model.load_state_dict(orig_state)
    model.eval()
    model = model.cuda()
    orig_metrics = calc_metrics(model, test_loader, device)
    step_cuttings = []
    sub_step_cuttings_list = []
    model.process_dataset_with_inner_data_extraction(test_loader)
    mids = []
    for step in range(num_of_layers):
        model = model.cpu()
        all_features, lowest_feature_value, size_value = model.recreation_with_filter_lowest_entropy_delete(step, 1)
        model.eval()
        model = model.cuda()
        exp_metrics = calc_metrics(model, test_loader, device)
        acc_drop = max((orig_metrics["acc"] - exp_metrics["acc"]), 0.000000001)
        step_cuttings.append((step, exp_metrics, all_features,
                              lowest_feature_value, size_value, acc_drop))
        model = get_new_model_instance()
        model.load_state_dict(orig_state)
        model.eval()
        sub_steps_cutting = []
        low = 0
        high = len(all_features)
        mid = 0
        while low <= high:
            mid = (high + low) // 2
            model.process_dataset_with_inner_data_extraction(test_loader)
            model = model.cpu()
            all_features, lowest_feature_value, size_value = model.recreation_with_filter_lowest_entropy_delete(step, mid)
            model.eval()
            model = model.cuda()
            exp_metrics = calc_metrics(model, test_loader, device)
            acc_drop = max((orig_metrics["acc"] - exp_metrics["acc"]), 0.000000001)
            sub_steps_cutting.append((step, exp_metrics, all_features,
                                      lowest_feature_value, size_value, acc_drop))
            model = get_new_model_instance()
            model.load_state_dict(orig_state)
            model.eval()

            if acc_drop < acceptable_loss_acc_value:
                low = mid + 1
            elif acc_drop > acceptable_loss_acc_value:
                high = mid - 1
            else:
                break
        mids.append(mid)
        sub_step_cuttings_list.append(sub_steps_cutting)

    model = model.cpu()
    acc_pool = 0.0
    results_cut = []
    for i, mid_num_v in enumerate(mids):
        mid_num = mid_num_v // num_of_layers
        if mid_num > 0:
            model.process_dataset_with_inner_data_extraction(test_loader)
            all_features, lowest_feature_value, size_value = model.recreation_with_filter_lowest_entropy_delete(i, mid_num)
            results_cut.append((all_features, lowest_feature_value, size_value, mid_num, i))
    model.eval()
    model = model.cuda()
    exp_metrics = calc_metrics(model, test_loader, device)
    result = {"exp_metrics": exp_metrics, "acc_pool": acc_pool, "results_cut": results_cut,
              "search": sub_step_cuttings_list, "mids": mids}
    return {"orig": orig_metrics, "steps": step_cuttings, "results": result}


def iterate_through_experiment_lowest_entropy_delete(directory_models: Path, directory_stats: Path, test_loader):
    directory_stats.mkdir(parents=True, exist_ok=True)
    models_paths = list(directory_models.glob("ep*.bin"))
    bef = len(models_paths)
    max_acc = max(map(lambda x: float(x.name[11:-4]), models_paths))
    models_paths = list(filter(lambda x: max_acc - float(x.name[11:-4]) <= acceptable_loss_acc_value, models_paths))
    aft = len(models_paths)
    print(f"Found: {bef}, calc: {aft}")
    for model_path in tqdm(models_paths, desc="models"):
        stats_output = directory_stats / (model_path.name[:-4] + ".pkl")
        if stats_output.exists():
            continue
        filter_metrics = experiment_on_model_with_lowest_filter_entropy(model_path, test_loader)
        result = {"filter": filter_metrics}
        with open(str(stats_output), "wb") as f:
            pickle.dump(result, f)


def iterate_through_hyperparams_lowest_entropy_delete(output_path: Path, batch_size=1024):
    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              test=True)
    for hyper_param in hyperparams_list:
        print(f"Work with hyperparams: {hyper_param.name}")
        for exp in hyper_param.glob("*"):
            print(f"Experiment: {exp.name}")
            output = output_path / hyper_param.name / exp.name
            iterate_through_experiment_lowest_entropy_delete(exp, output, test_loader)


if __name__ == "__main__":
    # batch_size = 4096
    # test_loader = data_loader(data_dir='./data',
    #                           batch_size=batch_size,
    #                           test=True)
    iterate_through_hyperparams_lowest_entropy_delete(output_path)
    # model_path = Path(
    #     "/home/kirrog/projects/FQWB/model/aug_4_block_reg_block/l1_0.0001_l2_1e-05_wd_1e-08/ep_049_acc_0.792200.bin")
    # experiment_on_model_with_lowest_block(model_path, test_loader)
# l1_1e-06_l2_1e-07_wd_1e-08 - ep_012
# l1_1e-05_l2_1e-06_wd_1e-08 - ep_024
