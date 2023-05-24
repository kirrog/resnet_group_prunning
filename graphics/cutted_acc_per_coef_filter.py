from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


def filter_cut_coefficient_graphic(data_metrics, name=""):
    parameters_cut_0_0 = []
    parameters_cut_0_1 = []
    parameters_cut_1_0 = []
    parameters_cut_1_1 = []
    parameters_best_cut_0_0 = []
    parameters_best_cut_0_1 = []
    parameters_best_cut_1_0 = []
    parameters_best_cut_1_1 = []
    coefficient = []
    for i, (k, v) in enumerate(sorted(data_metrics.items(), key=lambda x: int(x[0][6:8]))):
        epoch_num = 0
        best_epoch_num = 0
        max_acc = 0
        max_acc_cut = 0
        max_threshold = 0
        max_cut_acc = 0
        max_threshold_best = 0
        recr = None
        for j, value in enumerate(v):
            recr = value[2]
            acc = value["filter"]["orig"]["acc"]
            cut_filter = list(filter(lambda x: x[1]["acc"] != acc, value["filter"]["sub_steps"]))
            cut_acc_obj_list = list(sorted(cut_filter, key=lambda x: x[1]["acc"]))
            cut_acc_obj = cut_acc_obj_list[-1]
            cut_acc = cut_acc_obj[1]["acc"]
            if max_cut_acc < cut_acc:
                max_cut_acc = cut_acc
                best_epoch_num = j
                max_threshold_best = cut_acc_obj[0]
            if acc > max_acc:
                max_acc = acc
                epoch_num = j
                max_acc_cut = cut_acc
                max_threshold = cut_acc_obj[0]
        filter(lambda x: x, recr[0][0])
        # recr[0][1]
        # recr[3][0]
        # recr[3][1]
        # parameters_cut.append(len())
        # parameters_best_cut.append(max_cut_acc)
        coefficient.append(i + 1)
        print(f"Exp: {i}, ep: {epoch_num}, max_acc: {max_acc}, ep: {best_epoch_num} best_cut_acc: {max_cut_acc}")
    # plt.plot(coefficient, parameters_cut, label="Cut")
    # plt.plot(coefficient, parameters_best_cut, label="Best")
    plt.legend(loc="lower right")
    plt.xlabel("experiment number")
    plt.ylabel("accuracy")
    plt.title(name)
    plt.show()


def parameters_cut_coefficient_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats"))
    filter_cut_coefficient_graphic(b, "Block")
    filter_cut_coefficient_graphic(g, "Filter")