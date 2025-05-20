import json
from collections import defaultdict
from pprint import pprint

from matplotlib import pyplot as plt

# dataset_name = "cifar10_"
dataset_name = "butterfly_"
with open(f"{dataset_name}prepared_cuttings_stats.json", "r", encoding="utf-8") as f:
    experiments_dict = json.load(f)

bar_val_list = []
color_list = []

init_type2color = {
    "raw": "yellow",
    "none": "red",
    "v1": "green",
    "v2": "blue"
}

weight_reg_func_name2color = {
    "none": "c",
    "radem_v2-inv": (1.0, 0.0, 0.0, 0.3),
    "radem_v2": (1.0, 0.0, 0.0, 1.0),
    "radem-inv": (0.0, 1.0, 0.0, 0.3),
    "radem": (0.0, 1.0, 0.0, 1.0),
    "entropy-inv": (0.0, 0.0, 1.0, 0.3),
    "entropy": (0.0, 0.0, 1.0, 1.0),
    "weights": 'k',
}

inner_reg_func_name2color = {
    "none": "c",
    "radem-inv": (1.0, 0.0, 0.0, 0.3),  # "red"
    "radem": (1.0, 0.0, 0.0, 1.0),  # "red"
    "entr-inv": (0.0, 0.0, 1.0, 0.3),  # "blue"
    "entr": (0.0, 0.0, 1.0, 1.0),  # "blue"
    "l1_l2": 'black',
}
inner_reg_order_list = [
    "none", "l1_l2", "entr", "entr-inv", "radem", "radem-inv"
]

increment_stats = defaultdict(float)

increment_stats_num = defaultdict(int)

init_type_choose = "all"
# init_type_choose = "none"
# init_type_choose = "v1"
# init_type_choose = "v2"

for exp_params, exp_stats in experiments_dict.items():
    (init_type, epoch, weight_reg_func_name,
     inner_reg_func_name, learn_orig_reg,
     reg_values) = exp_params.split("___")
    del_num__list, acc__list, epoch_lowest_acc = exp_stats

    if init_type_choose != "all" and init_type != init_type_choose:
        continue

    acc_incr_speed_list = []
    prev_acc = epoch_lowest_acc[0]
    max_acc = prev_acc
    max_pos = 1
    prev_del_num = 0
    for i, (acc_, del_num) in enumerate(zip(acc__list, del_num__list)):
        cur_del_num = del_num - prev_del_num
        prev_del_num = del_num
        acc_incr = acc_ - prev_acc
        prev_acc = acc_
        speed = acc_incr / cur_del_num
        acc_incr_speed_list.append(speed)
        if acc_ > max_acc:
            max_acc = acc_
            max_pos = i
    acc_inc_speed_grow_list = acc_incr_speed_list[:max_pos]
    if len(acc_inc_speed_grow_list) == 0:
        continue

    # color_list.append(inner_reg_func_name2color[inner_reg_func_name])
    # bar_val_list.append((sum(acc_inc_speed_grow_list) / len(acc_inc_speed_grow_list)))

    # increment_stats[inner_reg_func_name] += del_num__list[-1]
    # increment_stats[inner_reg_func_name] += sum(acc_inc_speed_grow_list) / len(acc_inc_speed_grow_list)
    increment_stats[inner_reg_func_name] += max(acc_inc_speed_grow_list)
    # increment_stats[inner_reg_func_name] += max_acc - epoch_lowest_acc[0]

    increment_stats_num[inner_reg_func_name] += 1

# plt.bar(list(range(len(bar_val_list))), bar_val_list, color=color_list)
# plt.ylabel("Accuracy")
# plt.xlabel("Weights deleted")
#
# plt.title(f"Params reg func compare for l1 l2")
# plt.show()
# plt.savefig(f"../graphics_data/cuttings_compare/init_strategy_compare.png")

incr_stats_list = list(increment_stats.items())
pprint([func_name for func_name, stats_val in incr_stats_list])



plt.bar(list(range(len(incr_stats_list))),
        [increment_stats[func_name] / increment_stats_num[func_name] for func_name in inner_reg_order_list],
        color=[inner_reg_func_name2color[func_name] for func_name in inner_reg_order_list])
# plt.legend([func_name for func_name, stats_val in incr_stats_list])
# plt.ylabel("Del num")
plt.ylabel("Mean max speed")
# plt.ylabel("Accuracy increase")

plt.xlabel("Inner reg function")

# plt.title(f"Mean deleted weights by method. Init: {init_type_choose}")
plt.title(f"Accuracy increase mean max speed. Init: {init_type_choose}")
# plt.title(f"Accuracy increase. Init: {init_type_choose}")
# plt.title(f"Accuracy increase mean max speed")
# plt.show()
# plt.savefig(f"/home/kirrog/projects/FQWB/graphics_data/methods_mean_delete/butterfly/init_{init_type_choose}.png")
# plt.savefig(f"/home/kirrog/projects/FQWB/graphics_data/methods_mean_delete/butterfly/acc_incr_init_{init_type_choose}.png")
plt.savefig(f"/home/kirrog/projects/FQWB/graphics_data/methods_mean_delete/butterfly/mean_speed.png")
