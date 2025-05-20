import json
from collections import defaultdict
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from matplotlib.pyplot import legend

dataset_name = "bloodcells"
stats_root_dir = Path(f"/home/kirrog/projects/FQWB/model/{dataset_name}_v1_pos_drop_0.0")

index_order_weight = [
    "none",
    "weight",
    "entr",
    "entr-inv",
    "radem",
    "radem-inv"
]

index_order_inner = [
    "none",
    "weights",
    "entropy",
    "entropy-inv",
    "radem",
    "radem-inv",
    "radem_v2",
    "radem_v2-inv"
]


def comb2delete_num(comb: List[int]) -> int:
    comb_filters = [64, 64, 512, 512]
    acc = 0
    for comb_num, comb_filters in zip(comb, comb_filters):
        acc += comb_num * comb_filters
    return acc


experiments_dict = dict()

for cut_exp_path in stats_root_dir.glob("*"):
    learn_reg = "none"
    learn_orig_reg = "none"
    reg_values = "none"
    splitted_exp_name = str(cut_exp_path.name).split("__")
    exp_name_params = str(cut_exp_path.name).split("___")[1]
    init_type = "raw"
    if len(splitted_exp_name) > 3:
        init_type = splitted_exp_name[3]
        learn_reg = splitted_exp_name[-2]
        reg_values = splitted_exp_name[-1]
        if reg_values != "":
            learn_orig_reg = learn_reg
            if learn_reg == "radem_v2-inv":
                learn_reg = "radem-inv"
            if learn_reg == "radem_v2":
                learn_reg = "radem"
            if learn_reg == "entropy":
                learn_reg = "entr"
            if learn_reg == "entropy-inv":
                learn_reg = "entr-inv"
            if learn_reg == "weights":
                learn_reg = "weight"
        else:
            init_type = "raw"
    epoch_lowest_acc = [1.0]
    for cutting_stats_path in cut_exp_path.glob("*.json"):
        init_acc = float(str(cutting_stats_path.name).split("__")[0].split("_")[-1])

        epoch_str = str(cutting_stats_path.name).split("__")[0].split("_")[-3]
        if epoch_str == "result":
            continue
        epoch = int(str(cutting_stats_path.name).split("__")[0].split("_")[-3])

        inner_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[1].split("_")[2:])
        weight_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[2].split("_")[2:])
        if weight_reg_func_name != inner_reg_func_name and inner_reg_func_name != "none" and weight_reg_func_name != "none":
            continue
        if learn_reg != weight_reg_func_name:
            continue

        with open(cutting_stats_path, "r", encoding="utf-8") as f:
            cutting_stats_data = json.load(f)
        # if len(cutting_stats_data.keys()) < 17:
        #     continue
        del_num__list = []
        acc__list = []

        next_stats = []

        counter = 0

        for comb, (cut_acc, cut_stats) in cutting_stats_data.items():
            comb_coefs = [int(x) for x in comb.split("_")]
            next_stats.append((cut_acc, comb_coefs))
            counter += 1
            if counter % 4 == 0:
                max_acc_comb = None
                max_acc_acc = 0.0
                for cut_acc_old, comb_old in next_stats:
                    if cut_acc_old > max_acc_acc:
                        max_acc_acc = cut_acc_old
                        max_acc_comb = comb_old
                comb_del = comb2delete_num(max_acc_comb)
                # comb_del = cut_stats["size_value"]
                next_stats = []

                if len(acc__list) > 0 and max_acc_acc < acc__list[0]:
                    continue
                del_num__list.append(comb_del)
                acc__list.append(max_acc_acc)

        if acc__list[0] < epoch_lowest_acc[0]:
            epoch_lowest_acc[0] = acc__list[0]

        if "weight" in weight_reg_func_name:
            weight_reg_func_name = "l1_l2"
        if "weight" in inner_reg_func_name:
            inner_reg_func_name = "l1_l2"
        # label = f"ep_{epoch}_{weight_reg_func_name}_{inner_reg_func_name}"
        label = f"init_type_{init_type}"
        # plt.plot(del_num__list, acc__list, label=label, color=init_type2color[init_type])
        # if init_type != "raw":
        #     continue
        params = [init_type, epoch, weight_reg_func_name, inner_reg_func_name, learn_orig_reg, reg_values]
        experiments_dict["___".join([str(x) for x in params])] = (del_num__list, acc__list, epoch_lowest_acc)

with open(f"./prepared_data/{dataset_name}_prepared_cuttings_stats.json", "w", encoding="utf-8") as f:
    json.dump(experiments_dict, f)
