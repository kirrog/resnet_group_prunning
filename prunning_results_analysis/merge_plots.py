import json
from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

results_path = list(Path("./results/acc_agreg").glob("*/*.json"))
results_ebm_path = list(Path("./results_ebm/acc_agreg").glob("*/*.json"))
output_path = Path("./results_merged")

type_name2dict = defaultdict(list)
for res_p in results_path:
    type_name = str(res_p.parent.name)
    type_name2dict[type_name].append(res_p)

for res_p in results_ebm_path:
    type_name = str(res_p.parent.name)
    type_name2dict[type_name].append(res_p)

labels_names_list = ["L1_L2", "default", "neg_entr", "ebm"]

for type_name, paths_list in type_name2dict.items():
    output_path_dir = output_path / type_name
    output_path_dir.mkdir(exist_ok=True, parents=True)
    dataset_name2paths_list = defaultdict(list)
    for path_ in paths_list:
        dataset_name2paths_list[str(path_.name)].append(path_)
    for dataset_name, dataset_paths_list in dataset_name2paths_list.items():
        output_plot_path = output_path_dir / f"{dataset_name}.png"
        labels_list = []
        values_list = []
        ylabel_ = ""
        title_ = ""
        for path in dataset_paths_list:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                labels_small_list, values_small_list, ylabel_, title_ = data
                labels_list.extend(labels_small_list)
                values_list.extend(values_small_list)
        fig, ax = plt.subplots()
        # bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
        bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        founded_values = []
        labels_names_new_list = [x for x in labels_names_list if x in labels_list]
        for label_name in labels_names_new_list:
            if label_name in labels_list:
                i = labels_list.index(label_name)
                founded_values.append(values_list[i])
        # ax.bar(labels_list, values_list, color=bar_colors)
        ax.bar(labels_names_new_list, founded_values, color=bar_colors)

        ax.set_ylabel(ylabel_)
        ax.set_title(title_)

        plt.savefig(output_plot_path)
        plt.close()
