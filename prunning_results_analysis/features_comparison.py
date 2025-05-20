import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas
from tqdm import tqdm

input_data_path = Path("/media/kirrog/Expansion/prunning_results_analysis_data")


def block_stats_classing(block_stats_list: List[Tuple[float, float]]):
    mse_list = np.array([x[0] for x in block_stats_list])
    mse_low_row = np.array(range(len(mse_list) - 1, -1, -1), dtype=np.float64)
    mse_low_row /= np.max(mse_low_row)

    mse_corr_coef = np.corrcoef(mse_list, mse_low_row)[0, 1]

    correlation_list = np.array([x[1] for x in block_stats_list])
    correlation_up_row = np.array(range(len(correlation_list)), dtype=np.float64)
    correlation_up_row /= np.max(correlation_up_row)

    corr_corr_coef = np.corrcoef(correlation_list, correlation_up_row)[0, 1]

    return mse_corr_coef, corr_corr_coef


print("Loading data")
dataset2methods_corr_coefs = defaultdict(list)
for dataset_case in input_data_path.glob("*"):
    for experiment_case in tqdm(list(dataset_case.glob("*")), desc=f"Processing {str(dataset_case.name)}"):
        for epoch_num in experiment_case.glob("*"):
            with open(epoch_num / "compare.json", "r") as f:
                compare = json.load(f)
            with open(epoch_num / "stats.json", "r") as f:
                stats_list = json.load(f)
            prunning_method2corr_coef = defaultdict(list)
            for stats in stats_list:
                (
                    acc_val, inner_reg, weights_reg,
                    block_stats_steps_comparison, block_lists,
                    deleted_features, steps_features_values
                ) = stats
                if len(block_stats_steps_comparison["0"]) == 0:
                    continue
                corr_coef_dict = dict()
                for k, block_list_stats in block_stats_steps_comparison.items():
                    mse_corr_coef, corr_corr_coef = block_stats_classing(block_list_stats)
                    corr_coef_dict[k] = (mse_corr_coef, corr_corr_coef)
                prunning_method2corr_coef[f"{weights_reg}_{inner_reg}"].append(corr_coef_dict)
            dataset2methods_corr_coefs[str(dataset_case.name)].append(prunning_method2corr_coef)

list_of_columns = [
    "0_mse_negative", "0_mse_neutral", "0_mse_positive",

    "1_mse_negative", "1_mse_neutral", "1_mse_positive",

    "2_mse_negative", "2_mse_neutral", "2_mse_positive",

    "3_mse_negative", "3_mse_neutral", "3_mse_positive",

    "0_corr_negative", "0_corr_neutral", "0_corr_positive",

    "1_corr_negative", "1_corr_neutral", "1_corr_positive",

    "2_corr_negative", "2_corr_neutral", "2_corr_positive",

    "3_corr_negative", "3_corr_neutral", "3_corr_positive"
]
print("Merging results")
for dataset_name, methods_corr_coefs_list in dataset2methods_corr_coefs.items():
    print(f"Working on: {dataset_name}")
    prunning_method2all_corr_coef = defaultdict(list)
    for prunning_method2corr_coef in methods_corr_coefs_list:
        for method_name, coef_list in prunning_method2corr_coef.items():
            prunning_method2all_corr_coef[method_name].extend(coef_list)
    dataset_methods_result_dict = dict()
    for method, coef_list in prunning_method2all_corr_coef.items():
        result_dict = defaultdict(int)
        block_class2amount = defaultdict(int)
        for corr_coef_dict_of_tuple in coef_list:
            for block_num, (mse_corr_coef, corr_corr_coef) in corr_coef_dict_of_tuple.items():
                mse_class = ""
                if mse_corr_coef > 0.5:
                    mse_class = "mse_positive"
                elif mse_corr_coef > -0.5:
                    mse_class = "mse_neutral"
                else:
                    mse_class = "mse_negative"

                result_dict[f"{block_num}_{mse_class}"] += 1
                block_class2amount[f"{block_num}_mse"] += 1

                corr_class = ""
                if corr_corr_coef > 0.5:
                    corr_class = "corr_positive"
                elif corr_corr_coef > -0.5:
                    corr_class = "corr_neutral"
                else:
                    corr_class = "corr_negative"

                result_dict[f"{block_num}_{corr_class}"] += 1
                block_class2amount[f"{block_num}_corr"] += 1
            final_result_dict = dict()
            for block_num_class, amount in result_dict.items():
                block_num_class_short = "_".join(list(block_num_class.split("_"))[:-1])
                final_result_dict[block_num_class] = amount / block_class2amount[block_num_class_short]
            dataset_methods_result_dict[method] = final_result_dict
    (pandas.DataFrame(dataset_methods_result_dict).T
     .to_excel(f"results/steps_comparison/step_features_compare_{dataset_name}.xlsx", columns=list_of_columns))
