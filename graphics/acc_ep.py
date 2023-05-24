from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


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
