from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


def extract_identical(path_orig, path_def, path_ent):
    orig = load_metrics(path_orig)
    defa = load_metrics(path_def)
    entr = load_metrics(path_ent)
    output_data = []
    names = ["Base", "Block", "Filter"]
    counter = 0
    for elem_o, elem_d, elem_e in zip(orig, defa, entr):
        for k in set(elem_o.keys()).intersection(elem_d.keys()).intersection(elem_e.keys()):
            reg_data, default_data, entropy_data = elem_o[k], elem_d[k], elem_e[k]
            best_acc = list(sorted((set(map(lambda x: x["filename"], reg_data))
                                    .intersection(map(lambda x: x["filename"], default_data))
                                    .intersection(map(lambda x: x["filename"], entropy_data))),
                                   key=lambda x: float(x.split("_")[3][:-4])))
            elem_o[k] = list(filter(lambda x: x["filename"] in best_acc, reg_data))
            elem_d[k] = list(filter(lambda x: x["filename"] in best_acc, default_data))
            elem_e[k] = list(filter(lambda x: x["filename"] in best_acc, entropy_data))
        output_data.append((elem_o, f"{names[counter]}_original_acc2cut"))
        output_data.append((elem_d, f"{names[counter]}_rademacher_acc2cut"))
        output_data.append((elem_e, f"{names[counter]}_entropy_acc2cut"))
        counter += 1
    return output_data


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
    # plt.show()
    plt.savefig(f"../graphics_data/acc_per_filter_all/{name}.png")
    plt.close()


def accuracy_cut_coefficient_graphics():
    Path("../graphics_data/acc_per_filter_all").mkdir(exist_ok=True, parents=True)
    data = extract_identical(Path("/home/kirrog/projects/FQWB/model/stats_pool"),
                             Path("/home/kirrog/projects/FQWB/model/stats_radamcher_default"),
                             Path("/home/kirrog/projects/FQWB/model/stats_radamcher_entropy"))
    for elements, label_name in data:
        accuracy_cut_coefficient_graphic(elements, label_name)


if __name__ == "__main__":
    accuracy_cut_coefficient_graphics()
