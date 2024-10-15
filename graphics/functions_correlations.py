from pathlib import Path

from matplotlib import pyplot as plt

from graphics.base import load_metrics


def weights_correlation(data_metrics, default_dict, entropy_dict, name=""):
    for k in set(data_metrics.keys()).intersection(default_dict.keys()).intersection(entropy_dict.keys()):
        reg_data, default_data, entropy_data = data_metrics[k], default_dict[k], entropy_dict[k]
        best_acc = list(sorted((set(map(lambda x: x["filename"], reg_data))
                                .intersection(map(lambda x: x["filename"], default_data))
                                .intersection(map(lambda x: x["filename"], entropy_data))),
                               key=lambda x: float(x.split("_")[3][:-4])))[-1]
        reg_case = list(filter(lambda x: x["filename"] == best_acc, reg_data))[0]
        def_case = list(filter(lambda x: x["filename"] == best_acc, default_data))[0]
        ent_case = list(filter(lambda x: x["filename"] == best_acc, entropy_data))[0]
        for step_reg, step_case, step_ent, label_num in zip(reg_case["filter"]["steps"],
                                                            def_case["filter"]["steps"],
                                                            ent_case["filter"]["steps"],
                                                            range(len(ent_case["filter"]["steps"]))):
            x = list(range(len(step_reg[2])))
            norm = max([elem[1] for elem in step_reg[2]])
            y = [elem[1] / norm for elem in step_reg[2]]
            xy = [(x, y) for x, y in sorted(zip(x, y), key=lambda elem: elem[1])]
            order = [x for x, y in xy]
            y_ordered = [y[o] for o in order]

            plt.plot(y_ordered, "o", label="Weights")

            norm_max = max([abs(elem[1]) for elem in step_case[2]])
            norm_min = min([abs(elem[1]) for elem in step_case[2]])
            y = [(abs(elem[1]) - norm_min) / (norm_max - norm_min) for elem in step_case[2]]
            y_ordered = [y[o] for o in order]

            plt.plot(y_ordered, "o", label="Rademacher")

            norm = max([elem[1] for elem in step_ent[2]])
            y = [elem[1] / norm for elem in step_ent[2]]
            y_ordered = [y[o] for o in order]

            plt.plot(y_ordered, "o", label="Entropy")

            plt.legend(loc="upper right", numpoints=1)
            plt.xlabel("Group number")
            plt.ylabel("Feature value")
            plt.title(f"Exp: {name} type: {k} ResBlock number: {label_num}")
            # plt.show()
            plt.savefig(f"../graphics_data/correlations/exp_{name}_type_{k}_num_{label_num}.png")
            plt.close()


def accuracy_cut_coefficient_graphics():
    Path("../graphics_data/correlations").mkdir(exist_ok=True, parents=True)
    o_p, b_p, g_p = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_pool"))
    o_d, b_d, g_d = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_radamcher_default"))
    o_e, b_e, g_e = load_metrics(Path("/home/kirrog/projects/FQWB/model/stats_radamcher_entropy"))
    weights_correlation(o_p, o_d, o_e, "Original")
    weights_correlation(b_p, b_d, b_e, "Block")
    weights_correlation(g_p, g_d, g_e, "Filter")


if __name__ == "__main__":
    accuracy_cut_coefficient_graphics()
