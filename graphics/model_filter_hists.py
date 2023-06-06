import pickle

from matplotlib import pyplot as plt


def one_model_weights_hist(model_stats_path):
    with open(model_stats_path, "rb") as f:
        data = pickle.load(f)
    filters_lists = []
    for i in data["filter"]["sub_steps"][0][2]:
        for j in i:
            filters_lists.append(list(map(lambda x: x[1], j)))
    for fl in filters_lists:
        plt.hist(fl)
        plt.xlabel("Mean weight value")
        plt.ylabel("Count")
        plt.show()


if __name__ == "__main__":
    one_model_weights_hist(
        "/home/kirrog/projects/FQWB/model/stats/aug_4_block_reg_group/l1_1e-04_l2_1e-05_wd_1e-08/ep_049_acc_0.794600.pkl")
