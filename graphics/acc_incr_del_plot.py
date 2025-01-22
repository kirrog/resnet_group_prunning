import json

from matplotlib import pyplot as plt

with open("prepared_cuttings_stats.json", "r", encoding="utf-8") as f:
    experiments_dict = json.load(f)

init_type2color = {
    "raw": "yellow",
    "none": "red",
    "v1": "green",
    "v2": "blue"
}

exp_init_types = ["No reg train", "No init", "50 ep", "500 ep"]

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

exp_weight_reg_types = ["None", "RadOrig-inv", "RadOrig", "RadCust-inv", "RadCust", "Entr-inv", "Entr", "L1-L2"]

inner_reg_func_name2color = {
    "none": "c",
    "radem-inv": (0.0, 1.0, 0.0, 0.3),
    "radem": (0.0, 1.0, 0.0, 1.0),
    "entr-inv": (0.0, 0.0, 1.0, 0.3),
    "entr": (0.0, 0.0, 1.0, 1.0),
    "l1_l2": 'black',
}

exp_inner_reg_types = ["None", "RadOrig-inv", "RadOrig", "Entr-inv", "Entr", "L1-L2"]

params_reg_func_name2color = {
    "1e-10_1e-09": (0.0, 1.0, 0.0, 1.0),
    "1e-09_1e-08": (0.0, 1.0, 0.0, 0.9),
    "1e-08_1e-07": (0.0, 1.0, 0.0, 0.8),
    "1e-07_1e-06": (0.0, 1.0, 0.0, 0.7),
    "1e-06_1e-05": (0.0, 1.0, 0.0, 0.6),
    "1e-05_0.0001": (0.0, 1.0, 0.0, 0.5),
}

exp_params_reg_types = inner_reg_func_name2color.keys()

# colors_list = init_type2color.values()
colors_list = inner_reg_func_name2color.values()

for color in colors_list:
    plt.plot([0], [0.86], color=color)

for exp_params, exp_stats in experiments_dict.items():
    (init_type, epoch, weight_reg_func_name,
     inner_reg_func_name, learn_orig_reg,
     reg_values) = exp_params.split("___")
    del_num__list, acc__list, epoch_lowest_acc = exp_stats

    if "rad" not in inner_reg_func_name:
        continue

    del_num_final_list = [0]
    acc_final_list = [epoch_lowest_acc[0]]

    max_acc = 0.0
    best_cut_acc = 0.0
    max_acc_cut = 0
    best_cut_cut = 0
    for del_num, acc in zip(del_num__list, acc__list):
        if max_acc < acc:
            max_acc = acc
            max_acc_cut = del_num
        if best_cut_cut < del_num:
            best_cut_acc = acc
            best_cut_cut = del_num

    del_num_final_list.append(max_acc_cut)
    del_num_final_list.append(best_cut_cut)
    acc_final_list.append(max_acc)
    acc_final_list.append(best_cut_acc)

    plt.plot(del_num_final_list, acc_final_list, color=inner_reg_func_name2color[inner_reg_func_name])
    # plt.plot(del_num__list, acc__list, color=init_type2color[init_type])
plt.ylabel("Accuracy")
plt.xlabel("Weights deleted")
# plt.legend(exp_init_types, labelcolor=init_type2color.values())
plt.legend(exp_inner_reg_types)
plt.title(f"Params reg func compare for l1 l2")

plt.show()
# plt.savefig(f"../graphics_data/cuttings_compare/init_strategy_compare.png")
