import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

figure(figsize=(8, 6), dpi=80)
prunning_stats_path_dir = Path("/home/kirrog/projects/itmo/FQWB/model_ebm")
output_path = Path("./results_ebm")
with open("./results/experiment_hyperparameters2compare_ebm.json", "r", encoding="utf-8") as f:
    experiment_hyperparameters2compare_list = json.load(f)


def pos2childs_forming(positions_list_data):
    pos2childs = defaultdict(set)
    for i in range(len(positions_list_data)):
        for j in range(len(positions_list_data)):
            if i == j:
                continue
            l_pos = positions_list_data[i]
            r_pos = positions_list_data[j]
            if (sum([(y - x == 1) for x, y in zip(l_pos, r_pos)]) == 1 and
                    sum([(y == x) for x, y in zip(l_pos, r_pos)]) == 3):
                pos2childs[data_keys_list[i]].add(data_keys_list[j])
    return pos2childs


def calc_min_mean_max_disp_range(data_array):
    min_value = float(np.min(data_array))
    max_value = float(np.max(data_array))
    return min_value, float(np.mean(data_array)), max_value, float(np.var(data_array)), float(max_value - min_value)


###################################

def calc_acc_min_mean_max_disp_range(ordered_list):
    acc_numpy = np.array([stats["acc"] for stats in ordered_list])
    return calc_min_mean_max_disp_range(acc_numpy)


def calc_loss_min_mean_max_disp_range(ordered_list):
    loss_numpy = np.array([stats["loss"] for stats in ordered_list])
    return calc_min_mean_max_disp_range(loss_numpy)


def calc_acc_diff_per_step_min_mean_max_disp_range(ordered_list):
    acc_list = [stats["acc"] for stats in ordered_list]
    min_acc = min(acc_list)
    prev_acc = min_acc
    acc_diff_list = []
    for acc in acc_list:
        acc_diff = acc - prev_acc
        prev_acc = acc
        acc_diff_list.append(acc_diff)
    return calc_min_mean_max_disp_range(np.array(acc_diff_list))


def calc_loss_diff_per_step_min_mean_max_disp_range(ordered_list):
    loss_list = [stats["loss"] for stats in ordered_list]
    min_loss = min(loss_list)
    prev_loss = min_loss
    loss_diff_list = []
    for loss in loss_list:
        loss_diff = loss - prev_loss
        prev_loss = loss
        loss_diff_list.append(loss_diff)
    return calc_min_mean_max_disp_range(np.array(loss_diff_list))


def calc_acc_diff_per_weight_min_mean_max_disp_range(ordered_list):
    acc_list = [stats["acc"] for stats in ordered_list]
    min_acc = min(acc_list)
    prev_acc = min_acc
    acc_diff_per_weight_list = []
    for stats in ordered_list:
        acc_diff_per_weight = (stats["acc"] - prev_acc) / stats["size_value"]
        prev_acc = stats["acc"]
        acc_diff_per_weight_list.append(acc_diff_per_weight)
    return calc_min_mean_max_disp_range(np.array(acc_diff_per_weight_list))


def calc_acc_diff_per_loss_diff_min_mean_max_disp_range(ordered_list):
    acc_list = [stats["acc"] for stats in ordered_list]
    min_acc = min(acc_list)
    prev_acc = min_acc
    acc_diff_per_weight_list = []
    for stats in ordered_list:
        acc_diff_per_weight = (stats["acc"] - prev_acc) / stats["loss"]
        prev_acc = stats["acc"]
        acc_diff_per_weight_list.append(acc_diff_per_weight)
    return calc_min_mean_max_disp_range(np.array(acc_diff_per_weight_list))


def calc_weight_per_acc_diff_min_mean_max_disp_range(ordered_list):
    acc_list = [stats["acc"] for stats in ordered_list]
    min_acc = min(acc_list)
    prev_acc = min_acc
    weight_per_acc_diff_list = []
    for stats in ordered_list:
        acc_diff = stats["acc"] - prev_acc
        weight_per_acc_diff = stats["size_value"] / (0.1 if acc_diff == 0.0 else acc_diff)
        prev_acc = stats["acc"]
        weight_per_acc_diff_list.append(weight_per_acc_diff)
    return calc_min_mean_max_disp_range(np.array(weight_per_acc_diff_list))


def calc_weight_per_loss_diff_min_mean_max_disp_range(ordered_list):
    loss_list = [stats["loss"] for stats in ordered_list]
    min_loss = min(loss_list)
    prev_loss = min_loss
    weight_per_loss_diff_list = []
    for stats in ordered_list:
        loss_diff = stats["loss"] - prev_loss
        weight_per_loss_diff = stats["size_value"] / (1.0 if loss_diff == 0.0 else loss_diff)
        prev_loss = stats["loss"]
        weight_per_loss_diff_list.append(weight_per_loss_diff)
    return calc_min_mean_max_disp_range(np.array(weight_per_loss_diff_list))


def calc_loss_diff_per_acc_diff_min_mean_max_disp_range(ordered_list):
    acc_list = [stats["acc"] for stats in ordered_list]
    loss_list = [stats["loss"] for stats in ordered_list]
    min_acc = min(acc_list)
    min_loss = min(loss_list)
    prev_acc = min_acc
    prev_loss = min_loss
    loss_diff_per_acc_diff_list = []
    for stats in ordered_list:
        acc_diff = (stats["acc"] - prev_acc)
        loss_diff_per_acc_diff = (stats["loss"] - prev_loss) / (0.1 if acc_diff == 0.0 else acc_diff)
        prev_loss = stats["loss"]
        prev_acc = stats["acc"]
        loss_diff_per_acc_diff_list.append(loss_diff_per_acc_diff)
    return calc_min_mean_max_disp_range(np.array(loss_diff_per_acc_diff_list))


def calc_loss_diff_per_weight_min_mean_max_disp_range(ordered_list):
    loss_list = [stats["loss"] for stats in ordered_list]
    min_loss = min(loss_list)
    prev_loss = min_loss
    loss_diff_per_weight_list = []
    for stats in ordered_list:
        loss_diff_per_weight = (stats["loss"] - prev_loss) / stats["size_value"]
        prev_loss = stats["loss"]
        loss_diff_per_weight_list.append(loss_diff_per_weight)
    return calc_min_mean_max_disp_range(np.array(loss_diff_per_weight_list))


def calc_deleted_weights_amount(ordered_list):
    return sum([stats["size_value"] for stats in ordered_list])


calc_dict_list = [
    ("acc", calc_acc_min_mean_max_disp_range),
    ("loss", calc_loss_min_mean_max_disp_range),
    ("acc_diff_per_step", calc_acc_diff_per_step_min_mean_max_disp_range),
    ("loss_diff_per_step", calc_loss_diff_per_step_min_mean_max_disp_range),
    ("acc_diff_per_weight", calc_acc_diff_per_weight_min_mean_max_disp_range),
    ("acc_diff_per_loss_diff", calc_acc_diff_per_loss_diff_min_mean_max_disp_range),
    ("weight_per_acc_diff", calc_weight_per_acc_diff_min_mean_max_disp_range),
    ("weight_per_loss_diff", calc_weight_per_loss_diff_min_mean_max_disp_range),
    ("loss_diff_per_acc_diff", calc_loss_diff_per_acc_diff_min_mean_max_disp_range),
    ("loss_diff_per_weight", calc_loss_diff_per_weight_min_mean_max_disp_range)
]

calc_unique_feature_list = [
    ("deleted_weights_amount", calc_deleted_weights_amount)
]


#     if len(label_list) == 0:
#         continue
def printing_diagrams(label2values_dict, ylabel_str, xlabel_str, output_dir_path, dataset_name):
    label2values_list = [x for x in sorted(label2values_dict.items(), key=lambda x: x[0])]
    label_list = [x[0] for x in label2values_list]
    values_list_list = [x[1] for x in label2values_list]

    number_of_bins = 20

    hist_range = (min(np.min(values_list) for values_list in values_list_list),
                  max(np.max(values_list) for values_list in values_list_list))
    binned_data_sets = [
        np.histogram(d, range=hist_range, bins=number_of_bins)[0]
        for d in values_list_list
    ]
    binned_maximums = np.max(binned_data_sets, axis=1)
    x_locations = np.arange(0, max(binned_maximums) * len(binned_maximums), max(binned_maximums))

    bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights / 2

    fig, ax = plt.subplots()
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(centers, binned_data, height=heights, left=lefts)

    ax.set_xticks(x_locations, label_list)
    ax.set_title(f"Dataset: {dataset_name}")
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel(xlabel_str)

    plt.savefig(output_dir_path / f"{dataset_name}.png")
    plt.close()


def print_compare_diagram(output_path, dataset_name, diagram_name, all_dict, prunned_dict):
    fig, ax = plt.subplots()

    # fruits = ['apple', 'blueberry', 'cherry', 'orange']
    # counts = [40, 100, 30, 55]
    # bar_labels = ['red', 'blue', '_red', 'orange']
    bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    keys_union = set(all_dict.keys()).union(set(prunned_dict.keys()))
    value_list = []
    labels_list = []

    for key in sorted(keys_union):
        ratio = prunned_dict[key] / all_dict[key]
        value_list.append(ratio)
        labels_list.append(key)

    ax.bar(labels_list, value_list, color=bar_colors)

    ax.set_ylabel('Доля успешных прореживаний')
    ax.set_title(f'{diagram_name}')

    plt.savefig(output_path / f"{dataset_name}.png")
    with open(output_path / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump([labels_list, value_list, 'Доля успешных прореживаний', f'{diagram_name}'], f)


for dataset_dir_path in prunning_stats_path_dir.glob("*"):
    dataset_name = str(dataset_dir_path.name)[:-16]
    if dataset_name == "breast":
        continue
    methods_stats_list = []
    processed_counter = 0
    method2all_amount = defaultdict(int)
    method2prunned_amount = defaultdict(int)

    for prunning_exp_dir in tqdm(list(dataset_dir_path.glob("*")), desc=f"Processing: {dataset_name}"):
        epoch2stats = defaultdict(list)
        prunning_exp_dir_splitted = str(prunning_exp_dir.name).split("__")
        init_form = prunning_exp_dir_splitted[-3]
        regularization_method = prunning_exp_dir_splitted[-2:]
        if regularization_method[0] in ["radem", "radem-inv"]:
            continue
        processed_counter += 1
        for prunning_stats_elem_path in list(prunning_exp_dir.glob("*.json")):
            init_ep, inner_reg, weights_reg, _ = list(str(prunning_stats_elem_path.name).split("__"))
            _, _, ep_num, _, acc_val = list(init_ep.split("_"))
            inner_reg = inner_reg[10:]
            weights_reg = weights_reg[11:]

            if (f"{regularization_method[0]}___"
                f"{regularization_method[1]}___"
                f"{inner_reg}___"
                f"{weights_reg}") not in experiment_hyperparameters2compare_list:
                continue

            with open(prunning_stats_elem_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                v[1]["name"] = k
            data_keys_list = list(data.keys())
            data_keys_list.append("0_0_0_0")
            positions_list = [[int(x) // d for x, d in zip(x.split("_"), [1, 1, 8, 8])] for x in data_keys_list]

            pos2childs = pos2childs_forming(positions_list)

            ordered_stats = []
            neighbors_stats = []
            cur_str = "0_0_0_0"

            while True:
                if cur_str not in pos2childs:
                    break

                childs_list = pos2childs[cur_str]
                max_childs = 0
                max_str = ""
                neigbours_current = []
                best_stats = None
                parent_nums = list(cur_str.split("_"))
                for child in childs_list:
                    childs_nums = list(child.split("_"))

                    difference_position = 0

                    for k in range(4):
                        if parent_nums[k] != childs_nums[k]:
                            difference_position = k

                    if difference_position > 1:
                        elems2del = 8
                    else:
                        elems2del = 1

                    step_acc, stats = data[child]

                    childs_amount = len(pos2childs[child])

                    if childs_amount > max_childs:
                        if best_stats is not None:
                            neigbours_current.append(best_stats)
                        max_childs = childs_amount
                        best_stats = stats
                        max_str = child
                    else:
                        neigbours_current.append(stats)

                if best_stats is not None:
                    ordered_stats.append(best_stats)
                neighbors_stats.append(neigbours_current)
                cur_str = max_str

            ordered_len = len(ordered_stats)
            neighbours_len = sum([len(x) for x in neighbors_stats])
            data_len = len(data)
            if ordered_len + neighbours_len != data_len:
                print(f"{ordered_len} + {neighbours_len} = {data_len} : {prunning_stats_elem_path}")

            reg_method_ = regularization_method[0]
            if reg_method_ == "radem_v2":
                reg_method_ = "radem"
            if reg_method_ == "radem_v2-inv":
                reg_method_ = "neg_radem"
            if reg_method_ == "entropy-inv":
                reg_method_ = "neg_entropy"
            if reg_method_ == "weight":
                reg_method_ = "L1_L2"
            if reg_method_ == "weights":
                reg_method_ = "L1_L2"
            if reg_method_ == "none":
                reg_method_ = "default"

            prune_inner_method_ = inner_reg
            if prune_inner_method_ == "entr-inv":
                prune_inner_method_ = "neg_entr"
            if prune_inner_method_ == "radem-inv":
                prune_inner_method_ = "neg_radem"
            if prune_inner_method_ == "weight":
                prune_inner_method_ = "L1_L2"
            if prune_inner_method_ == "none":
                prune_inner_method_ = "default"

            prune_weight_method_ = weights_reg
            if prune_weight_method_[0] == "_":
                prune_weight_method_ = prune_weight_method_[1:]
            if prune_weight_method_ == "entr-inv":
                prune_weight_method_ = "neg_entr"
            if prune_weight_method_ == "radem-inv":
                prune_weight_method_ = "neg_radem"
            if prune_weight_method_ == "weight":
                prune_weight_method_ = "L1_L2"
            if prune_weight_method_ == "none":
                prune_weight_method_ = "default"

            if len(ordered_stats) <= 1:
                method2all_amount[f"{reg_method_}___"
                                  f"{prune_inner_method_}___"
                                  f"{prune_weight_method_}"] += 1
                continue
            else:
                method2all_amount[f"{reg_method_}___"
                                  f"{prune_inner_method_}___"
                                  f"{prune_weight_method_}"] += 1
                method2prunned_amount[f"{reg_method_}___"
                                      f"{prune_inner_method_}___"
                                      f"{prune_weight_method_}"] += 1

            stat_func_results = dict()
            stat_func_results["init_form"] = init_form
            for stat_name, stat_func in calc_dict_list:
                stat_min, stat_mean, stat_max, stat_disp, stat_range = stat_func(ordered_stats)
                stat_func_results[f"{stat_name}_stat_min"] = stat_min
                stat_func_results[f"{stat_name}_stat_mean"] = stat_mean
                stat_func_results[f"{stat_name}_stat_max"] = stat_max
                stat_func_results[f"{stat_name}_stat_disp"] = stat_disp
                stat_func_results[f"{stat_name}_stat_range"] = stat_range
            for stat_name, stat_func in calc_unique_feature_list:
                stat_func_results[stat_name] = stat_func(ordered_stats)

            stat_func_results["reg_method"] = reg_method_
            stat_func_results["reg_params"] = regularization_method[1]

            stat_func_results["prune_inner_method"] = prune_inner_method_

            stat_func_results["prune_weight_method"] = prune_weight_method_

            methods_stats_list.append(stat_func_results)

    reg_method_output_path = output_path / "acc_agreg" / "inner_deleted_mean"
    reg_method_output_path.mkdir(exist_ok=True, parents=True)
    reg_method_params_dict = defaultdict(list)
    for stat_elem in methods_stats_list:
        reg_method_params_dict[stat_elem["prune_inner_method"]].append(stat_elem["deleted_weights_amount"])

    value_list = []
    labels_list = []
    for inner_method_name, deleted_weights_list in sorted(reg_method_params_dict.items(), key=lambda x: x[0]):
        value_list.append(sum(deleted_weights_list) / len(deleted_weights_list))
        labels_list.append(inner_method_name)
    fig, ax = plt.subplots()
    bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    ax.bar(labels_list, value_list, color=bar_colors)

    ax.set_ylabel('Среднее число удаленных весов')
    ax.set_title(f'Сравнение оценок активаций')

    plt.savefig(reg_method_output_path / f"{dataset_name}.png")
    plt.close()
    with open(reg_method_output_path / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump([labels_list, value_list, 'Среднее число удаленных весов', f'Сравнение оценок активаций'], f)





    reg_method_output_path = output_path / "acc_agreg" / "weight_deleted_mean"
    reg_method_params_dict = defaultdict(list)
    reg_method_output_path.mkdir(exist_ok=True, parents=True)
    for stat_elem in methods_stats_list:
        reg_method_params_dict[stat_elem["prune_weight_method"]].append(stat_elem["deleted_weights_amount"])

    value_list = []
    labels_list = []
    for inner_method_name, deleted_weights_list in sorted(reg_method_params_dict.items(), key=lambda x: x[0]):
        value_list.append(sum(deleted_weights_list) / len(deleted_weights_list))
        labels_list.append(inner_method_name)
    fig, ax = plt.subplots()
    bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    ax.bar(labels_list, value_list, color=bar_colors)

    ax.set_ylabel('Среднее число удаленных весов')
    ax.set_title(f'Сравнение оценок параметров')

    plt.savefig(reg_method_output_path / f"{dataset_name}.png")
    plt.close()
    with open(reg_method_output_path / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump([labels_list, value_list, 'Среднее число удаленных весов', f'Сравнение оценок параметров'], f)







    reg_method_dict_all = defaultdict(int)
    reg_method_dict_prunned = defaultdict(int)
    prune_inner_method_dict_all = defaultdict(int)
    prune_inner_method_dict_prunned = defaultdict(int)
    prune_weight_method_dict_all = defaultdict(int)
    prune_weight_method_dict_prunned = defaultdict(int)

    keys_ = set(method2prunned_amount.keys()).union(set(method2all_amount.keys()))

    for key in keys_:
        reg_method, prune_inner_method_, prune_weight_method_ = key.split("___")

        reg_method_dict_all[reg_method] += method2all_amount[key]
        prune_inner_method_dict_all[prune_inner_method_] += method2all_amount[key]
        prune_weight_method_dict_all[prune_weight_method_] += method2all_amount[key]

        reg_method_dict_prunned[reg_method] += method2prunned_amount[key]
        prune_inner_method_dict_prunned[prune_inner_method_] += method2prunned_amount[key]
        prune_weight_method_dict_prunned[prune_weight_method_] += method2prunned_amount[key]

    ratio_output_path = output_path / "acc_agreg"

    reg_method_ratio_output_path = ratio_output_path / "reg_method"
    reg_method_ratio_output_path.mkdir(exist_ok=True, parents=True)
    print_compare_diagram(reg_method_ratio_output_path, dataset_name, "Сравнение регуляризаций",
                          reg_method_dict_all,
                          reg_method_dict_prunned)

    reg_method_ratio_output_path = ratio_output_path / "prune_inner"
    reg_method_ratio_output_path.mkdir(exist_ok=True, parents=True)
    print_compare_diagram(reg_method_ratio_output_path, dataset_name, "Сравнение оценок активаций",
                          prune_inner_method_dict_all,
                          prune_inner_method_dict_prunned)

    reg_method_ratio_output_path = ratio_output_path / "prune_weight"
    reg_method_ratio_output_path.mkdir(exist_ok=True, parents=True)
    print_compare_diagram(reg_method_ratio_output_path, dataset_name, "Сравнение оценок параметров",
                          prune_weight_method_dict_all,
                          prune_weight_method_dict_prunned)

# print(f"Processed counter: {processed_counter}")
# # reg_method - labels - reg_method
# reg_method_output_path = output_path / "reg_method"
# reg_method_params_dict = defaultdict(list)
# for stat_elem in methods_stats_list:
#     reg_method_params_dict[stat_elem["reg_method"]].append(stat_elem)
#
# for stat_name, _ in calc_dict_list:
#     reg_method_stat_name_output_dir_path = reg_method_output_path / stat_name
#     for stat in ["stat_min", "stat_mean", "stat_max", "stat_disp", "stat_range"]:
#         stat_reg_method_stat_name_output_dir_path = reg_method_stat_name_output_dir_path / stat
#         stat_reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#
#         label2values_dict = dict()
#         for label, values_list in reg_method_params_dict.items():
#             label2values_dict[label] = np.array([x[f"{stat_name}_{stat}"] for x in values_list])
#
#         if len(label2values_dict) == 0:
#             continue
#         printing_diagrams(label2values_dict,
#                           f"Score: {stat_name} : {stat}",
#                           "Regularization methods",
#                           stat_reg_method_stat_name_output_dir_path, dataset_name)
#
# for stat_name, _ in calc_unique_feature_list:
#     reg_method_stat_name_output_dir_path = reg_method_output_path / stat_name
#     reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#     label2values_dict = dict()
#     for label, values_list in reg_method_params_dict.items():
#         label2values_dict[label] = np.array([x[stat_name] for x in values_list])
#
#     if len(label2values_dict) == 0:
#         continue
#     printing_diagrams(label2values_dict,
#                       f"Score: {stat_name}",
#                       "Regularization methods",
#                       reg_method_stat_name_output_dir_path, dataset_name)
#
# # reg_method - labels - reg_params for each method
# reg_method_output_path = output_path / "reg_params"
# reg_method_params_dict = defaultdict(list)
# methods_set = set()
# for stat_elem in methods_stats_list:
#     methods_set.add(stat_elem["reg_method"])
#     reg_method_params_dict[stat_elem["reg_params"]].append(stat_elem)
#
# for method_name in methods_set:
#     reg_method_param_stat_name_output_dir_path = reg_method_output_path / method_name
#     for stat_name, _ in calc_dict_list:
#         reg_method_stat_name_output_dir_path = reg_method_param_stat_name_output_dir_path / stat_name
#         for stat in ["stat_min", "stat_mean", "stat_max", "stat_disp", "stat_range"]:
#             stat_reg_method_stat_name_output_dir_path = reg_method_stat_name_output_dir_path / stat
#             stat_reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#
#             label2values_dict = dict()
#             for label, values_list in reg_method_params_dict.items():
#                 reg_method_elements_list = [x[f"{stat_name}_{stat}"] for x in values_list if
#                                             x["reg_method"] == method_name]
#                 if len(reg_method_elements_list) == 0:
#                     continue
#                 label2values_dict[label] = np.array(reg_method_elements_list)
#
#             if len(label2values_dict) == 0:
#                 continue
#             printing_diagrams(label2values_dict,
#                               f"Score: {stat_name} : {stat}",
#                               f"Regularization of {method_name} method parameters",
#                               stat_reg_method_stat_name_output_dir_path, dataset_name)
#
#     for stat_name, _ in calc_unique_feature_list:
#         reg_method_stat_name_output_dir_path = reg_method_param_stat_name_output_dir_path / stat_name
#         reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#         label2values_dict = dict()
#         for label, values_list in reg_method_params_dict.items():
#             label2values_dict[label] = np.array([x[stat_name] for x in values_list])
#
#         if len(label2values_dict) == 0:
#             continue
#         printing_diagrams(label2values_dict,
#                           f"Score: {stat_name}",
#                           f"Regularization of {method_name} method parameters",
#                           reg_method_stat_name_output_dir_path, dataset_name)
#
# # prune_inner_method - labels - prune_inner_method
# reg_method_output_path = output_path / "inner_method"
# reg_method_params_dict = defaultdict(list)
# for stat_elem in methods_stats_list:
#     reg_method_params_dict[stat_elem["prune_inner_method"]].append(stat_elem)
#
# for stat_name, _ in calc_dict_list:
#     reg_method_stat_name_output_dir_path = reg_method_output_path / stat_name
#     for stat in ["stat_min", "stat_mean", "stat_max", "stat_disp", "stat_range"]:
#         stat_reg_method_stat_name_output_dir_path = reg_method_stat_name_output_dir_path / stat
#         stat_reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#
#         label2values_dict = dict()
#         for label, values_list in reg_method_params_dict.items():
#             label2values_dict[label] = np.array([x[f"{stat_name}_{stat}"] for x in values_list])
#
#         if len(label2values_dict) == 0:
#             continue
#         printing_diagrams(label2values_dict,
#                           f"Score: {stat_name} : {stat}",
#                           "Prune inner methods",
#                           stat_reg_method_stat_name_output_dir_path, dataset_name)
#
# for stat_name, _ in calc_unique_feature_list:
#     reg_method_stat_name_output_dir_path = reg_method_output_path / stat_name
#     reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#     label2values_dict = dict()
#     for label, values_list in reg_method_params_dict.items():
#         label2values_dict[label] = np.array([x[stat_name] for x in values_list])
#
#     if len(label2values_dict) == 0:
#         continue
#     printing_diagrams(label2values_dict,
#                       f"Score: {stat_name}",
#                       "Prune inner methods",
#                       reg_method_stat_name_output_dir_path, dataset_name)
#
# # prune_weight_method - labels - prune_weight_method
# reg_method_output_path = output_path / "weight_method"
# reg_method_params_dict = defaultdict(list)
# for stat_elem in methods_stats_list:
#     reg_method_params_dict[stat_elem["prune_weight_method"]].append(stat_elem)
#
# for stat_name, _ in calc_dict_list:
#     reg_method_stat_name_output_dir_path = reg_method_output_path / stat_name
#     for stat in ["stat_min", "stat_mean", "stat_max", "stat_disp", "stat_range"]:
#         stat_reg_method_stat_name_output_dir_path = reg_method_stat_name_output_dir_path / stat
#         stat_reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#
#         label2values_dict = dict()
#         for label, values_list in reg_method_params_dict.items():
#             label2values_dict[label] = np.array([x[f"{stat_name}_{stat}"] for x in values_list])
#
#         if len(label2values_dict) == 0:
#             continue
#         printing_diagrams(label2values_dict,
#                           f"Score: {stat_name} : {stat}",
#                           "Prune weight methods",
#                           stat_reg_method_stat_name_output_dir_path, dataset_name)
#
# for stat_name, _ in calc_unique_feature_list:
#     reg_method_stat_name_output_dir_path = reg_method_output_path / stat_name
#     reg_method_stat_name_output_dir_path.mkdir(exist_ok=True, parents=True)
#     label2values_dict = dict()
#     for label, values_list in reg_method_params_dict.items():
#         label2values_dict[label] = np.array([x[stat_name] for x in values_list])
#
#     if len(label2values_dict) == 0:
#         continue
#     printing_diagrams(label2values_dict,
#                       f"Score: {stat_name}",
#                       "Prune weight methods",
#                       reg_method_stat_name_output_dir_path, dataset_name)
