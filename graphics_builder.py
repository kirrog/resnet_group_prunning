import pickle
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def load_exp(path2metrics: Path):
    results = dict()
    for hyps in path2metrics.glob("*"):
        pickles_data = []
        for ep in list(sorted(list(hyps.glob("*")), key=lambda x: int(x.name[3:6]))):
            with open(str(ep), "rb") as f:
                data = pickle.load(f)
                pickles_data.append(data)
        results[str(hyps.name)] = pickles_data
    return results


def load_metrics(path2metrics: Path):
    orig = path2metrics / "aug_4_block"
    block = path2metrics / "aug_4_block_reg_block"
    group = path2metrics / "aug_4_block_reg_group"
    orig_metrics = load_exp(orig)
    block_metrics = load_exp(block)
    group_metrics = load_exp(group)
    return orig_metrics, block_metrics, group_metrics


def accuracy_epochs_graphic(data_metrics, name=""):
    for i, (k, v) in enumerate(sorted(data_metrics.items(), key=lambda x: int(x[0][6:8]))):
        metrics = []
        epoch_num = 0
        max_acc = 0
        for j, value in enumerate(v):
            acc = value["filter"]["orig"]["acc"]
            if acc > max_acc:
                max_acc = acc
                epoch_num = j
            metrics.append(acc)
        print(f"Exp: {i}, ep: {epoch_num}, max_acc: {max_acc}")
        plt.plot(list(range(len(metrics))), metrics, label=f"{i + 1}")
    plt.legend(loc="lower right")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(name)
    plt.show()


def accuracy_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats"))
    accuracy_epochs_graphic(o, "Original")
    accuracy_epochs_graphic(b, "Block")
    accuracy_epochs_graphic(g, "Filter")


def accuracy_cut_coefficient_graphic(data_metrics, name=""):
    accuracy_orig = []
    accuracy_cut = []
    accuracy_best_cut = []
    coefficient = []
    for i, (k, v) in enumerate(sorted(data_metrics.items(), key=lambda x: int(x[0][6:8]))):
        epoch_num = 0
        best_epoch_num = 0
        max_acc = 0
        max_acc_cut = 0
        max_cut_acc = 0
        for j, value in enumerate(v):
            acc = value["filter"]["orig"]["acc"]
            cut_filter = list(filter(lambda x: x[1]["acc"] != acc, value[name.lower()]["sub_steps"]))
            # cut_filter = list(filter(lambda x: x[1]["acc"] != acc, value["block"]["sub_steps"]))
            # cut_filter = list(filter(lambda x: x[1]["acc"] != acc, value["filter"]["sub_steps"]))
            cut_acc_obj_list = list(sorted(cut_filter, key=lambda x: x[1]["acc"]))
            cut_acc_obj = cut_acc_obj_list[-1]
            cut_acc = cut_acc_obj[1]["acc"]
            if max_cut_acc < cut_acc:
                max_cut_acc = cut_acc
                best_epoch_num = j
            if acc > max_acc:
                max_acc = acc
                epoch_num = j
                max_acc_cut = cut_acc
        accuracy_orig.append(max_acc)
        accuracy_cut.append(max_acc_cut)
        accuracy_best_cut.append(max_cut_acc)
        coefficient.append(i + 1)
        print(f"Exp: {i}, ep: {epoch_num}, max_acc: {max_acc}, ep: {best_epoch_num} best_cut_acc: {max_cut_acc}")
    plt.plot(coefficient, accuracy_orig, label="Original")
    plt.plot(coefficient, accuracy_cut, label="Cut")
    plt.plot(coefficient, accuracy_best_cut, label="Best")
    plt.legend(loc="lower right")
    plt.xlabel("experiment number")
    plt.ylabel("accuracy")
    plt.title(name)
    plt.show()


def accuracy_cut_coefficient_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats"))
    accuracy_cut_coefficient_graphic(b, "Block")
    accuracy_cut_coefficient_graphic(g, "Filter")


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
    #     parameters_cut.append(len())
    #     parameters_best_cut.append(max_cut_acc)
    #     coefficient.append(i + 1)
    #     print(f"Exp: {i}, ep: {epoch_num}, max_acc: {max_acc}, ep: {best_epoch_num} best_cut_acc: {max_cut_acc}")
    # plt.plot(coefficient, parameters_cut, label="Cut")
    # plt.plot(coefficient, parameters_best_cut, label="Best")
    # plt.legend(loc="lower right")
    # plt.xlabel("experiment number")
    # plt.ylabel("accuracy")
    # plt.title(name)
    # plt.show()


def parameters_cut_coefficient_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats"))
    accuracy_cut_coefficient_graphic(b, "Block")
    accuracy_cut_coefficient_graphic(g, "Filter")


def one_model_weights_hist(model_stats_path):
    with open(model_stats_path, "rb") as f:
        data = pickle.load(f)
    filters_lists = []
    for i in data["filter"]["sub_steps"][0][2]:
        for j in i:
            filters_lists.append(list(map(lambda x: x[1], j)))
    for fl in filters_lists:
        plt.hist(fl)
        plt.show()


if __name__ == "__main__":
    accuracy_cut_coefficient_graphics()
