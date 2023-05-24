from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


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
            cut_acc = value["filter"]["results"]["exp_metrics"]["acc"]
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
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_pool"))
    accuracy_cut_coefficient_graphic(b, "Block")
    accuracy_cut_coefficient_graphic(g, "Filter")


if __name__ == "__main__":
    accuracy_cut_coefficient_graphics()
