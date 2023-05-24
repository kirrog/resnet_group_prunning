from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


def acc_per_weight_graphic(data_metrics, name="", model_num_of_weights=14505354):
    parameters_best_cut_0_0 = []
    parameters_best_cut_0_1 = []
    parameters_best_cut_1_0 = []
    parameters_best_cut_1_1 = []
    coefficient = []
    for i, (exp_name, epochs) in enumerate(sorted(data_metrics.items(), key=lambda x: int(x[0][6:8]))):
        max_cut_acc_0 = 0
        best_epoch_num_0 = 0
        max_cut_acc_1 = 0
        best_epoch_num_1 = 0
        max_cut_acc_2 = 0
        best_epoch_num_2 = 0
        max_cut_acc_3 = 0
        best_epoch_num_3 = 0
        for j, value in enumerate(epochs):
            cut_block_0 = value["block"]["steps"][0][1]["acc"]
            cut_block_1 = value["block"]["steps"][1][1]["acc"]
            cut_block_2 = value["block"]["steps"][2][1]["acc"]
            cut_block_3 = value["block"]["steps"][3][1]["acc"]
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

        parameters_best_cut_0_0.append((epochs[best_epoch_num_0]["block"]["orig"]["acc"] - max_cut_acc_0) / (
                    model_num_of_weights - epochs[best_epoch_num_0]["block"]["steps"][0][2]))
        parameters_best_cut_0_1.append((epochs[best_epoch_num_1]["block"]["orig"]["acc"] - max_cut_acc_1) / (
                    model_num_of_weights - epochs[best_epoch_num_1]["block"]["steps"][1][2]))
        parameters_best_cut_1_0.append((epochs[best_epoch_num_2]["block"]["orig"]["acc"] - max_cut_acc_2) / (
                    model_num_of_weights - epochs[best_epoch_num_2]["block"]["steps"][2][2]))
        parameters_best_cut_1_1.append((epochs[best_epoch_num_3]["block"]["orig"]["acc"] - max_cut_acc_3) / (
                    model_num_of_weights - epochs[best_epoch_num_3]["block"]["steps"][3][2]))
        coefficient.append(i + 1)

    plt.plot(coefficient, parameters_best_cut_0_0, label="Best_0")
    plt.plot(coefficient, parameters_best_cut_0_1, label="Best_1")
    plt.plot(coefficient, parameters_best_cut_1_0, label="Best_2")
    plt.plot(coefficient, parameters_best_cut_1_1, label="Best_3")
    plt.legend(loc="lower right")
    plt.xlabel("experiment number")
    plt.ylabel("accuracy")
    plt.title(name)
    plt.show()


def acc_per_weight_coefficient_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_block"))
    acc_per_weight_graphic(b, "Block")
    acc_per_weight_graphic(g, "Filter")


if __name__ == "__main__":
    acc_per_weight_coefficient_graphics()
