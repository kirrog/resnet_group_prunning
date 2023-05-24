from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


def block_cut_acc_graphic(data_metrics, name=""):
    acc_values = []
    parameters_cut_0_0 = []
    parameters_cut_0_1 = []
    parameters_cut_1_0 = []
    parameters_cut_1_1 = []
    parameters_best_cut_0_0 = []
    parameters_best_cut_0_1 = []
    parameters_best_cut_1_0 = []
    parameters_best_cut_1_1 = []
    coefficient = []
    for i, (exp_name, epochs) in enumerate(sorted(data_metrics.items(), key=lambda x: int(x[0][6:8]))):
        epoch_num = 0
        max_acc = 0
        best_epoch_num = 0
        max_cut_acc = 0
        max_acc_cut = [0] * 4
        max_cut_acc_0 = 0
        best_epoch_num_0 = 0
        max_cut_acc_1 = 0
        best_epoch_num_1 = 0
        max_cut_acc_2 = 0
        best_epoch_num_2 = 0
        max_cut_acc_3 = 0
        best_epoch_num_3 = 0
        for j, value in enumerate(epochs):
            acc = value["block"]["orig"]["acc"]
            cut_block_0 = value["block"]["steps"][0][1]["acc"]
            cut_block_1 = value["block"]["steps"][1][1]["acc"]
            cut_block_2 = value["block"]["steps"][2][1]["acc"]
            cut_block_3 = value["block"]["steps"][3][1]["acc"]
            if max_acc < acc:
                max_acc = acc
                epoch_num = j
                max_acc_cut = [cut_block_0, cut_block_1, cut_block_2, cut_block_3]
            if max_cut_acc < max(cut_block_0, cut_block_1, cut_block_2, cut_block_3):
                max_cut_acc = max(cut_block_0, cut_block_1, cut_block_2, cut_block_3)
                best_epoch_num = j
            if max_cut_acc_0 < cut_block_0:
                max_cut_acc_0 = cut_block_0
                best_epoch_num_0 = j
            if max_cut_acc_1 < cut_block_1:
                max_cut_acc_1 = cut_block_1
                best_epoch_num_1 = j
            if max_cut_acc_2 < cut_block_2:
                max_cut_acc_2 = cut_block_2
                best_epoch_num_2 = j
            if max_cut_acc_3 < cut_block_3:
                max_cut_acc_3 = cut_block_3
                best_epoch_num_3 = j
        acc_values.append(max_acc)
        parameters_cut_0_0.append(epochs[epoch_num]["block"]["steps"][0][1]["acc"])
        parameters_cut_0_1.append(epochs[epoch_num]["block"]["steps"][1][1]["acc"])
        parameters_cut_1_0.append(epochs[epoch_num]["block"]["steps"][2][1]["acc"])
        parameters_cut_1_1.append(epochs[epoch_num]["block"]["steps"][3][1]["acc"])
        parameters_best_cut_0_0.append(max_cut_acc_0)
        parameters_best_cut_0_1.append(max_cut_acc_1)
        parameters_best_cut_1_0.append(max_cut_acc_2)
        parameters_best_cut_1_1.append(max_cut_acc_3)
        coefficient.append(i + 1)
        print(f"Exp: {i}, ep: {epoch_num}, max_acc: {max_acc}, ep: {best_epoch_num} best_cut_acc: {max_cut_acc}")
    plt.plot(coefficient, acc_values, label="Orig")
    plt.plot(coefficient, parameters_cut_0_0, label="Cut_0")
    plt.plot(coefficient, parameters_cut_0_1, label="Cut_1")
    plt.plot(coefficient, parameters_cut_1_0, label="Cut_2")
    plt.plot(coefficient, parameters_cut_1_1, label="Cut_3")
    plt.plot(coefficient, parameters_best_cut_0_0, label="Best_0")
    plt.plot(coefficient, parameters_best_cut_0_1, label="Best_1")
    plt.plot(coefficient, parameters_best_cut_1_0, label="Best_2")
    plt.plot(coefficient, parameters_best_cut_1_1, label="Best_3")
    plt.legend(loc="lower right")
    plt.xlabel("experiment number")
    plt.ylabel("accuracy")
    plt.title(name)
    plt.show()


def parameters_cut_coefficient_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_block"))
    block_cut_acc_graphic(b, "Block")
    block_cut_acc_graphic(g, "Filter")


if __name__ == "__main__":
    parameters_cut_coefficient_graphics()
