from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


def accuracy_cut_coefficient_graphic(data_metrics, name=""):
    accuracy = []
    cutted_filters = []
    labels = []
    for i, (k, v) in enumerate(sorted(data_metrics.items(), key=lambda x: float(x[0].split("_")[1]))):
        acc = []
        filt = []
        for j, value in enumerate(v):
            cut_acc = value["filter"]["results"]["exp_metrics"]["acc"]
            cut_weights = sum([x * (y // 4) for x, y in
                               zip([576, 576, 4608, 4608], value["filter"]["results"]["mids"])])

            # cut_weights = sum([(x[2]) * (y // 4) for x, y in
            #                    zip(value["filter"]["results"]["results_cut"], value["filter"]["results"]["mids"])])
            acc.append(cut_acc)
            filt.append(cut_weights)
        labels.append(i + 1)
        accuracy.append(acc)
        cutted_filters.append(filt)
    for acc, filt, label in zip(accuracy, cutted_filters, labels):
        plt.plot(filt, acc, "o", label=label)
    plt.legend(loc="upper right", numpoints=1)
    plt.xlabel("cutted weights")
    plt.ylabel("accuracy")
    plt.title(name)
    plt.show()


def accuracy_cut_coefficient_graphics():
    o, b, g = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_pool"))
    accuracy_cut_coefficient_graphic(o, "Original")
    accuracy_cut_coefficient_graphic(b, "Block")
    accuracy_cut_coefficient_graphic(g, "Filter")


if __name__ == "__main__":
    accuracy_cut_coefficient_graphics()
