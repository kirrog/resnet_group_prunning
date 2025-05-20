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
    dataset_name = str(dataset_case.name)
    methods2mse_amount__list = [defaultdict(float) for _ in range(4)]
    methods2mse_count__list = [defaultdict(int) for _ in range(4)]
    methods2correlation_count_pos = [defaultdict(int) for _ in range(4)]
    methods2correlation_count_neu = [defaultdict(int) for _ in range(4)]
    methods2correlation_count_neg = [defaultdict(int) for _ in range(4)]
    for experiment_case in tqdm(list(dataset_case.glob("*")), desc=f"Processing {str(dataset_case.name)}"):
        for epoch_num in experiment_case.glob("*"):
            with open(epoch_num / "compare.json", "r") as f:
                compare = json.load(f)
            with open(epoch_num / "stats.json", "r") as f:
                stats_list = json.load(f)
            prunning_method2corr_coef = defaultdict(list)
            num2method_name = dict()
            for i, stats in enumerate(stats_list):
                (
                    acc_val, inner_reg, weights_reg,
                    block_stats_steps_comparison, block_lists,
                    deleted_features, steps_features_values
                ) = stats
                num2method_name[str(i)] = "_".join(f"{weights_reg}_{inner_reg}".split("_")).strip("_")

            for i_j, comparison_list in compare.items():
                compare_data = comparison_list[0]
                i, j = i_j.split("_")
                for block_num, (mse, corr_coef) in compare_data.items():
                    methods2mse_amount__list[int(block_num)][f"{num2method_name[i]}___{num2method_name[j]}"] += mse
                    methods2mse_count__list[int(block_num)][f"{num2method_name[i]}___{num2method_name[j]}"] += 1
                    methods2mse_amount__list[int(block_num)][f"{num2method_name[j]}___{num2method_name[i]}"] += mse
                    methods2mse_count__list[int(block_num)][f"{num2method_name[j]}___{num2method_name[i]}"] += 1

                    if corr_coef > 0.5:
                        methods2correlation_count_pos[int(block_num)][
                            f"{num2method_name[j]}___{num2method_name[i]}"] += 1
                    elif corr_coef > -0.5:
                        methods2correlation_count_neu[int(block_num)][
                            f"{num2method_name[j]}___{num2method_name[i]}"] += 1
                    else:
                        methods2correlation_count_neg[int(block_num)][
                            f"{num2method_name[j]}___{num2method_name[i]}"] += 1
    page_counter = 0
    with pandas.ExcelWriter(f"results/methods_correlations/methods_mse_per_block_{dataset_name}.xlsx") as f:
        methods2mse_amount_dict_result_all = defaultdict(dict)
        for methods2mse_amount_dict, methods2mse_count_dict in zip(methods2mse_amount__list, methods2mse_count__list):
            methods2mse_amount_dict_result = defaultdict(dict)
            for methods, mse_amount in methods2mse_amount_dict.items():
                method_l, method_r = list(str(methods).split("___"))
                methods2mse_amount_dict_result[method_l][method_r] = mse_amount / methods2mse_count_dict[methods]
                if method_r not in methods2mse_amount_dict_result_all[method_l]:
                    methods2mse_amount_dict_result_all[method_l][method_r] = 0.0
                methods2mse_amount_dict_result_all[method_l][method_r] += mse_amount / methods2mse_count_dict[methods]
            block_data = pandas.DataFrame(methods2mse_amount_dict_result)
            block_data = block_data.sort_index(axis=0)
            block_data = block_data.sort_index(axis=1)
            block_data.to_excel(f, sheet_name=f"Block_{page_counter}")
            page_counter += 1
        block_data = pandas.DataFrame(methods2mse_amount_dict_result_all)
        block_data = block_data.sort_index(axis=0)
        block_data = block_data.sort_index(axis=1)
        block_data.to_excel(f, sheet_name=f"Block_sum")

    page_counter = 0
    with pandas.ExcelWriter(f"results/methods_correlations/methods_corr_classes_per_block_{dataset_name}.xlsx") as f:
        methods2mse_amount_dict_result_all = defaultdict(dict)
        methods2stats_tuple = dict()
        for methods2correlation_count_pos_dict, methods2correlation_count_neu_dict, methods2correlation_count_neg_dict in zip(
                methods2correlation_count_pos, methods2correlation_count_neu, methods2correlation_count_neg):
            methods2mse_amount_dict_result = defaultdict(dict)
            for methods in methods2correlation_count_pos_dict.keys():
                method_l, method_r = list(str(methods).split("___"))
                count_pos = methods2correlation_count_pos_dict[methods]
                count_neu = methods2correlation_count_neu_dict[methods]
                count_neg = methods2correlation_count_neg_dict[methods]
                elems_list = [count_pos, count_neu, count_neg]
                methods2mse_amount_dict_result[method_l][method_r] = "/".join(
                    [f"{x / sum(elems_list):.04f}" for x in elems_list])
                if methods not in methods2stats_tuple:
                    methods2stats_tuple[methods] = [0.0, 0.0, 0.0]
                methods2stats_tuple[methods][0] += count_pos
                methods2stats_tuple[methods][1] += count_neu
                methods2stats_tuple[methods][2] += count_neg
            block_data = pandas.DataFrame(methods2mse_amount_dict_result)
            block_data = block_data.sort_index(axis=0)
            block_data = block_data.sort_index(axis=1)
            block_data.to_excel(f, sheet_name=f"Block_{page_counter}")
            page_counter += 1
        methods2mse_amount_dict_result_all_ = defaultdict(dict)
        for methods in methods2stats_tuple.keys():
            method_l, method_r = list(str(methods).split("___"))
            count_pos = methods2stats_tuple[methods][0]
            count_neu = methods2stats_tuple[methods][1]
            count_neg = methods2stats_tuple[methods][2]
            elems_list = [count_pos, count_neu, count_neg]
            methods2mse_amount_dict_result_all_[method_l][method_r] = "/".join(
                [f"{x / sum(elems_list):.04f}" for x in elems_list])
        block_data = pandas.DataFrame(methods2mse_amount_dict_result_all_)
        block_data = block_data.sort_index(axis=0)
        block_data = block_data.sort_index(axis=1)
        block_data.to_excel(f, sheet_name=f"Block_sum")
