import json
from collections import defaultdict
from pathlib import Path

import pandas

stats_root_dir = Path("/home/kirrog/projects/FQWB/model/v1_pos_drop_0.0")

cuttable_stat = defaultdict(dict)
uncuttable_stat = defaultdict(dict)

for cut_exp_path in stats_root_dir.glob("*"):
    for cutting_stats_path in cut_exp_path.glob("*.json"):
        inner_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[1].split("_")[2:])
        weight_reg_func_name = "_".join(str(cutting_stats_path.name).split("__")[2].split("_")[2:])
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

uncuttable_stats_data = pandas.DataFrame(uncuttable_stat)
print(uncuttable_stats_data)
cuttable_stats_data = pandas.DataFrame(cuttable_stat)
print(cuttable_stats_data)

final_stat = defaultdict(dict)
for inner_reg_func_name, v in uncuttable_stat.items():
    for weight_reg_func_name, amount in v.items():
        cuttable_amount = cuttable_stats_data[inner_reg_func_name][weight_reg_func_name]
        final_stat[inner_reg_func_name][weight_reg_func_name] = cuttable_amount / (cuttable_amount + amount)

final_stats_data = pandas.DataFrame(final_stat)
print(final_stats_data)
final_stats_data.to_excel("final_stats.xlsx")
