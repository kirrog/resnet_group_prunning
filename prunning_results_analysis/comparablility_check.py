import json
from collections import defaultdict
from pathlib import Path

prunning_stats_path_dir = Path("/home/kirrog/projects/FQWB/model")

dataset_elements_set_list = []
for dataset_dir_path in prunning_stats_path_dir.glob("*"):
    dataset_name = str(dataset_dir_path.name)[:-16]
    if dataset_name == "breast":
        continue
    dataset_elements_set = set()
    dataset_elements_set_list.append(dataset_elements_set)
    methods_stats_list = []
    for prunning_exp_dir in list(dataset_dir_path.glob("*")):
        epoch2stats = defaultdict(list)
        regularization_method = str(prunning_exp_dir.name).split("__")[-2:]
        if regularization_method[0] in ["radem", "radem-inv"]:
            continue
        for prunning_stats_elem_path in list(prunning_exp_dir.glob("*.json")):
            init_ep, inner_reg, weights_reg, _ = list(str(prunning_stats_elem_path.name).split("__"))
            _, _, ep_num, _, acc_val = list(init_ep.split("_"))
            inner_reg = inner_reg[10:]
            weights_reg = weights_reg[11:]
            dataset_elements_set.add(
                f"{regularization_method[0]}___{regularization_method[1]}___{inner_reg}___{weights_reg}")

current_dataset_set = None
for dataset_elements_set in dataset_elements_set_list:
    if current_dataset_set is None:
        current_dataset_set = dataset_elements_set
    else:
        current_dataset_set = current_dataset_set.intersection(dataset_elements_set)
with open("./results/experiment_hyperparameters2compare.json", "w", encoding="utf-8") as f:
    json.dump(list(current_dataset_set), f, ensure_ascii=False)
