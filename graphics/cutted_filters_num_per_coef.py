from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from graphics.base import load_metrics


def cutted_filters_num_per_coef(data_metrics, name=""):
    accuracy_best_cut = []
    best_mids_list = []
    coefficient = []
    for i, (k, v) in enumerate(sorted(data_metrics.items(), key=lambda x: int(x[0][6:8]))):
        best_epoch_num = 0
        max_cut_acc = 0
        best_mids = [0] * 4
        for j, value in enumerate(v):
            cut_acc = value["filter"]["results"]["exp_metrics"]["acc"]
            if max_cut_acc < cut_acc:
                max_cut_acc = cut_acc
                best_epoch_num = j
                best_mids = value["filter"]["results"]["mids"]
        accuracy_best_cut.append(max_cut_acc)
        best_mids_list.append(best_mids)
        coefficient.append(i + 1)
        print(f"Exp: {i}, ep: {best_epoch_num} best_cut_acc: {max_cut_acc}")

    weight_counts = {
        "Block_0": list(map(lambda x: x[0] // 4, best_mids_list)),
        "Block_1": list(map(lambda x: x[1] // 4, best_mids_list)),
        "Block_2": list(map(lambda x: x[2] // 4, best_mids_list)),
        "Block_3": list(map(lambda x: x[3] // 4, best_mids_list))
    }
    width = 0.5

    bottom = np.zeros(len(coefficient))

    for boolean, weight_count in weight_counts.items():
        plt.bar(coefficient, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    plt.legend(loc="upper right")
    plt.xlabel("experiment number")
    plt.ylabel("number of cut")
    plt.title(name)
    plt.show()


def cutted_per_coef_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_pool"))
    cutted_filters_num_per_coef(b, "Block")
    cutted_filters_num_per_coef(g, "Filter")


if __name__ == "__main__":
    cutted_per_coef_graphics()
