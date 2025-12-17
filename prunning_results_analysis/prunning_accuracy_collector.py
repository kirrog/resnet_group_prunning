from pathlib import Path

import pandas
import matplotlib.pyplot as plt

models_root_dir = Path("/media/kirrog/Expansion/models")
result_dir = Path("./results/acc_agreg")

dataset_name2params = {
    "d_cifar10": (10, 60000),
    "d_cifar100": (100, 60000),
    "d_butterfly": (75, 6499),
    "d_crop": (22, 25220),
    "d_bloodcells": (17, 17092),
    "d_fourniture": (32, 12360),
    "d_tinyimagenet": (200, 120000)
}

default_init = []
v1_init = []
v2_init = []

with pandas.ExcelFile(result_dir / f"acc_table.xlsx") as f:
    for sheet_name in f.sheet_names:
        sheet_dataframe = f.parse(sheet_name)
        if not sheet_name in dataset_name2params:
            continue
        params = dataset_name2params[sheet_name]
        # print(sheet_dataframe)

        for i, row in sheet_dataframe.iterrows():
            l1l2 = row["L1 и L2"]
            init = row["Иниц."]
            if init is not None:
                init = 0.0
            neg_radem = row["Нег. радем."]
            neg_entropy = row["Нег. энтропия"]
            radem = row["Радем."]
            entropy = row["Энтропия"]
            if i == 0:
                default_init.append((max([l1l2, init, neg_radem, neg_entropy, radem, entropy]), params[0], params[1]))
            if i == 1:
                v1_init.append((max([l1l2, init, neg_radem, neg_entropy, radem, entropy]), params[0], params[1]))
            if i == 2:
                v2_init.append((max([l1l2, init, neg_radem, neg_entropy, radem, entropy]), params[0], params[1]))

    default_init = list(sorted(default_init, key=lambda x: x[1]))
    v1_init = list(sorted(v1_init, key=lambda x: x[1]))
    v2_init = list(sorted(v2_init, key=lambda x: x[1]))
    plt.plot([x[1] for x in default_init], [x[0] for x in default_init], color="red", label="Без иниц.")
    plt.plot([x[1] for x in v1_init], [x[0] for x in v1_init], color="green", label="50 иниц.")
    plt.plot([x[1] for x in v2_init], [x[0] for x in v2_init], color="blue", label="500 иниц.")
    plt.ylabel('Точность')
    plt.xlabel('Число категорий')
    plt.legend()
    plt.savefig("./results/acc2cat.png")
    plt.close()

    default_init = list(sorted(default_init, key=lambda x: x[2]))
    v1_init = list(sorted(v1_init, key=lambda x: x[2]))
    v2_init = list(sorted(v2_init, key=lambda x: x[2]))
    plt.plot([x[2] for x in default_init], [x[0] for x in default_init], color="red", label="Без иниц.")
    plt.plot([x[2] for x in v1_init], [x[0] for x in v1_init], color="green", label="50 иниц.")
    plt.plot([x[2] for x in v2_init], [x[0] for x in v2_init], color="blue", label="500 иниц.")
    plt.ylabel('Точность')
    plt.xlabel('Число изображений')
    plt.legend()
    plt.savefig("./results/acc2img.png")
    plt.close()
