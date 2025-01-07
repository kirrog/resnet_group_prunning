import json
from collections import defaultdict
from pathlib import Path

import pandas

stats_root_dir = Path("/home/kirrog/projects/FQWB/model/v1_pos_drop_0.0")

index_order = [
    "none",
    "weight",
    "entr",
    "entr-inv",
    "radem",
    "radem-inv"
]

cuttable_stat = defaultdict(dict)
uncuttable_stat = defaultdict(dict)
debug_list = list()
for cut_exp_path in stats_root_dir.glob("*"):
    learn_reg = "none"
    splitted_exp_name = str(cut_exp_path.name).split("__")
    if len(splitted_exp_name) > 3:
        learn_reg = splitted_exp_name[-2]
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
        inner_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[1].split("_")[2:])
        weight_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[2].split("_")[2:])
        if weight_reg_func_name != inner_reg_func_name and inner_reg_func_name != "none" and weight_reg_func_name != "none":
            continue
        if learn_reg != weight_reg_func_name:
            continue
        debug_dict[f"{learn_reg}_{weight_reg_func_name}_{inner_reg_func_name}"] += 1
        with open(cutting_stats_path, "r", encoding="utf-8") as f:
            cutting_stats_data = json.load(f)
        if len(cutting_stats_data.keys()) < 17:
            if weight_reg_func_name not in uncuttable_stat[inner_reg_func_name]:
                uncuttable_stat[inner_reg_func_name][weight_reg_func_name] = 0
            uncuttable_stat[inner_reg_func_name][weight_reg_func_name] += 1
        else:
            if weight_reg_func_name not in cuttable_stat[inner_reg_func_name]:
                cuttable_stat[inner_reg_func_name][weight_reg_func_name] = 0
            cuttable_stat[inner_reg_func_name][weight_reg_func_name] += 1
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

amount_stats_data = pandas.DataFrame(amount_stat)
print(amount_stats_data)
final_stats_data = pandas.DataFrame(final_stat)
print(final_stats_data)
# final_stats_data.sort_index(axis=1).sort_index(axis=0).to_excel("final_stats.xlsx")
final_stats_data.reindex(index_order, axis=1).reindex(index_order, axis=0).to_excel("final_stats.xlsx")
