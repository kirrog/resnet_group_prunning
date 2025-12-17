from collections import defaultdict
from pathlib import Path
from pprint import pprint

import pandas
from pandas import DataFrame

models_root_dir = Path("/media/kirrog/Expansion/models")
result_dir = Path("./results/acc_agreg")

dataset2dicts_stats = defaultdict(list)
for model_exp in models_root_dir.glob("*"):
    model_exp_features = str(model_exp.name).split("___")
    base_name = model_exp_features[1]
    base_name_list = list(base_name.split("__"))
    dataset_name = base_name_list[0]
    init_type_ = base_name_list[1]
    reg_method_ = base_name_list[2]
    reg_coefs_ = base_name_list[3]
    if len(dataset_name.split("_")) > 1:
        dataset_name = list(dataset_name.split("_"))[0]
        init_type_ = "v2"
        reg_method_ = "none"

    if init_type_ == "none" and reg_method_ == "none":
        init_type_ = "v1"

    best_acc = 0.0
    best_ep = 0
    for model_inst in model_exp.glob("*.bin"):
        splitted_name = list(str(model_inst.name).split("_"))
        ep = splitted_name[1]
        int_ep = 0
        if ep.isnumeric():
            int_ep = int(ep)

        exp_acc = float(splitted_name[-1][:-4])
        if best_acc < exp_acc:
            best_acc = exp_acc
            best_ep = int_ep

    name_features = list(base_name.split("__"))
    if len(name_features) == 1:
        print(f"ACHTUNG: {model_exp}")
    else:
        dataset2dicts_stats[dataset_name].append({"coefs": reg_coefs_, "init": init_type_, "reg_method": reg_method_,
                                                  "stats": {"acc": best_acc, "best": best_ep}})
with pandas.ExcelWriter(result_dir / f"acc_table.xlsx") as f:
    for dataset_name, dataset_stats in dataset2dicts_stats.items():
        stats = defaultdict(dict)
        for stat in dataset_stats:
            init = stat["init"]
            if init == "none":
                init = "Без иниц."
            if init == "v1":
                init = "Иниц. 50 эп"
            if init == "v2":
                init = "Иниц. 500 эп"

            reg_method = stat["reg_method"]
            if reg_method == "entropy":
                reg_method = "Энтропия"
            if reg_method == "entropy-inv":
                reg_method = "Нег. энтропия"
            if reg_method == "radem_v2":
                reg_method = "Радем."
            if reg_method == "radem_v2-inv":
                reg_method = "Нег. радем."
            if reg_method == "weights":
                reg_method = "L1 и L2"
            if reg_method == "none":
                reg_method = "Иниц."

            acc = stat["stats"]["acc"]
            if init in stats[reg_method]:
                if stats[reg_method][init] < acc:
                    stats[reg_method][init] = acc
            else:
                stats[reg_method][init] = acc
        print(f"Dataset: {dataset_name}")
        print(DataFrame(stats))

        block_data = pandas.DataFrame(stats)
        block_data = block_data.sort_index(axis=0)
        block_data = block_data.sort_index(axis=1)
        block_data.to_excel(f, sheet_name=f"d_{dataset_name}")
