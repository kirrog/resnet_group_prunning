import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from matplotlib import pyplot as plt

init_type2color = {
    "raw": "yellow",
    "none": "red",
    "v1": "green",
    "v2": "blue"
}

order_list = ["raw", "none", "v1", "v2"]
init_type_choose_list = ["all", "none", "v1", "v2"]
dataset_name_list = [
    "butterfly", "cifar10", "cifar100", "crop", "flowers", "fourniture", "tinyimagenet", "vehicle", "bloodcells"
]

for dataset_name in dataset_name_list:
    with open(f"./prepared_data/{dataset_name}_prepared_cuttings_stats.json", "r", encoding="utf-8") as f:
        experiments_dict = json.load(f)
    Path(f"/home/kirrog/projects/FQWB/graphics_data/init_compare/{dataset_name}").mkdir(parents=True, exist_ok=True)
    sum_acc_stats = defaultdict(float)
    sum_acc_stats_num = defaultdict(int)
    for exp_params, exp_stats in experiments_dict.items():
        (init_type, epoch, weight_reg_func_name,
         inner_reg_func_name, learn_orig_reg,
         reg_values) = exp_params.split("___")
        del_num__list, acc__list, epoch_lowest_acc = exp_stats

        # if init_type_choose != "all" and init_type != init_type_choose:
        #     continue

        prev_acc = epoch_lowest_acc[0]
        max_acc = prev_acc
        max_del_num = 0
        max_pos = 0
        for i, (acc_, del_num) in enumerate(zip(acc__list, del_num__list)):
            if acc_ > max_acc:
                max_acc = acc_
                max_pos = i
        del_num_max_acc = del_num__list[max_pos]

        # sum_acc_stats[init_type] += max_acc - 0.8
        sum_acc_stats[init_type] += del_num_max_acc
        sum_acc_stats_num[init_type] += 1

    incr_stats_list = list(sum_acc_stats.items())
    pprint([func_name for func_name, stats_val in incr_stats_list])

    plt.bar(list(range(len(incr_stats_list))),
            [sum_acc_stats[name] / (sum_acc_stats_num[name] + 0.00001) for name in order_list],
            color=[init_type2color[func_name] for func_name in order_list],
            label=order_list)
    plt.legend(order_list)
    plt.ylabel("Mean del num")
    plt.xlabel("Init type")

    plt.title(f"Mean del num per pretrain type")
    # plt.show()
    plt.savefig(
        f"/home/kirrog/projects/FQWB/graphics_data/init_compare/{dataset_name}/mean_del_num.png")
    plt.close()
