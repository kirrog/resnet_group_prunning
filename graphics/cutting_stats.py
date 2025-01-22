import json
from collections import defaultdict
from pathlib import Path

import pandas
from pandas import ExcelWriter

stats_root_dir = Path("/home/kirrog/projects/FQWB/model/v1_pos_drop_0.0")

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


def comb2delete_num(comb: str) -> int:
    comb_filters = [64, 64, 512, 512]
    acc = 0
    for comb_num, comb_filters in zip([int(x) for x in comb.split("_")], comb_filters):
        acc += comb_num * comb_filters
    return acc


cuttable_stat = defaultdict(dict)
uncuttable_stat = defaultdict(dict)
max_incr_acc_stat = defaultdict(dict)
best_incr_acc_stat = defaultdict(dict)

train_init_comparation = {
    "none": defaultdict(dict),
    "v1": defaultdict(dict),
    "v2": defaultdict(dict)
}

debug_list = list()

for cut_exp_path in stats_root_dir.glob("*"):
    learn_reg = "none"
    learn_orig_reg = "none"
    splitted_exp_name = str(cut_exp_path.name).split("__")
    dataset_name = "none"
    if len(splitted_exp_name) > 3:
        dataset_name = splitted_exp_name[3]
        learn_reg = splitted_exp_name[-2]
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
    debug_dict = defaultdict(int)
    for cutting_stats_path in cut_exp_path.glob("*.json"):
        init_acc = float(str(cutting_stats_path.name).split("__")[0].split("_")[-1])
        inner_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[1].split("_")[2:])
        weight_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[2].split("_")[2:])
        if weight_reg_func_name != inner_reg_func_name and inner_reg_func_name != "none" and weight_reg_func_name != "none":
            continue
        if learn_reg != weight_reg_func_name:
            continue
        debug_dict[f"{learn_reg}_{weight_reg_func_name}_{inner_reg_func_name}"] += 1
        with open(cutting_stats_path, "r", encoding="utf-8") as f:
            cutting_stats_data = json.load(f)

        max_cut_acc = init_acc
        max_cut_comb = "0_0_0_0"
        max_cut_del = 0
        best_cut_acc = init_acc
        best_cut_comb = "0_0_0_0"
        best_cut_del = 0
        for comb, (cut_acc, _) in cutting_stats_data.items():
            comb_del = comb2delete_num(comb)
            if max_cut_acc <= cut_acc:
                max_cut_acc = cut_acc
                max_cut_comb = comb
            if best_cut_del < comb_del and init_acc <= cut_acc:
                best_cut_acc = cut_acc
                best_cut_del = comb_del

        if learn_orig_reg not in max_incr_acc_stat[inner_reg_func_name]:
            max_incr_acc_stat[inner_reg_func_name][learn_orig_reg] = (
                max_cut_acc, max_cut_acc - init_acc, max_cut_comb, max_cut_del)
        else:
            n_mca, n_dif, n_comb, n_del = max_incr_acc_stat[inner_reg_func_name][learn_orig_reg]
            if n_dif < max_cut_acc - init_acc:
                max_incr_acc_stat[inner_reg_func_name][learn_orig_reg] = (
                    max_cut_acc, max_cut_acc - init_acc, max_cut_comb, max_cut_del)

        if learn_orig_reg not in best_incr_acc_stat[inner_reg_func_name]:
            best_incr_acc_stat[inner_reg_func_name][learn_orig_reg] = (
                best_cut_acc, best_cut_acc - init_acc, best_cut_comb, best_cut_del)
        else:
            n_mca, n_dif, n_comb, n_del = best_incr_acc_stat[inner_reg_func_name][learn_orig_reg]
            if n_del < best_cut_del:
                best_incr_acc_stat[inner_reg_func_name][learn_orig_reg] = (
                    best_cut_acc, best_cut_acc - init_acc, best_cut_comb, best_cut_del)

        if len(cutting_stats_data.keys()) < 17:
            if learn_orig_reg not in uncuttable_stat[inner_reg_func_name]:
                uncuttable_stat[inner_reg_func_name][learn_orig_reg] = 0
            uncuttable_stat[inner_reg_func_name][learn_orig_reg] += 1
        else:
            if learn_orig_reg not in cuttable_stat[inner_reg_func_name]:
                cuttable_stat[inner_reg_func_name][learn_orig_reg] = 0
            cuttable_stat[inner_reg_func_name][learn_orig_reg] += 1
    debug_list.append(max(debug_dict.values()) if len(debug_dict) > 0 else 0)
print(f"Debug list: {debug_list}")
uncuttable_stats_data = pandas.DataFrame(uncuttable_stat)
print(uncuttable_stats_data)
cuttable_stats_data = pandas.DataFrame(cuttable_stat)
print(cuttable_stats_data)

final_stat = defaultdict(dict)
amount_stat = defaultdict(dict)
for inner_reg_func_name, v in uncuttable_stat.items():
    for weight_reg_func_name, amount in v.items():
        cuttable_amount = cuttable_stats_data[inner_reg_func_name][weight_reg_func_name]
        final_stat[inner_reg_func_name][weight_reg_func_name] = cuttable_amount / (cuttable_amount + amount)
        amount_stat[inner_reg_func_name][weight_reg_func_name] = cuttable_amount + amount

with ExcelWriter("final_stats.xlsx") as f:
    amount_stats_data = pandas.DataFrame(amount_stat)
    print(amount_stats_data)
    final_stats_data = pandas.DataFrame(final_stat)
    print(final_stats_data)
    final_stats_data.reindex(index_order_inner, axis=0).reindex(index_order_weight, axis=1).to_excel(f,
                                                                                                     sheet_name="ratio")

    max_incr_acc_stats_data = pandas.DataFrame(max_incr_acc_stat)
    print(max_incr_acc_stats_data)

    best_incr_acc_stats_data = pandas.DataFrame(best_incr_acc_stat)
    print(best_incr_acc_stats_data)

    for i, name in zip(range(4), ["acc", "dir", "comb", "del"]):
        data = defaultdict(dict)
        for k, v in max_incr_acc_stat.items():
            for k_, v_ in v.items():
                data[k][k_] = v_[i]
        pandas_data = pandas.DataFrame(data)
        print(pandas_data)
        pandas_data.reindex(index_order_inner, axis=0).reindex(index_order_weight, axis=1).to_excel(f,
                                                                                                    sheet_name=f"max_incr_{name}")
    for i, name in zip(range(4), ["acc", "dir", "comb", "del"]):
        data = defaultdict(dict)
        for k, v in best_incr_acc_stat.items():
            for k_, v_ in v.items():
                data[k][k_] = v_[i]
        pandas_data = pandas.DataFrame(data)
        print(pandas_data)
        pandas_data.reindex(index_order_inner, axis=0).reindex(index_order_weight, axis=1).to_excel(f,
                                                                                                    sheet_name=f"best_cut{name}")
