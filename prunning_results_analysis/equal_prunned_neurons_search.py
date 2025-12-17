import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

prunning_stats_path_dir = Path("/home/kirrog/projects/FQWB/model")
with open("./results/experiment_hyperparameters2compare.json", "r", encoding="utf-8") as f:
    experiment_hyperparameters2compare_list = json.load(f)


def pos_nums_list2str(l: List[int]) -> str:
    return "_".join([str(x) for x in l])


output_path = Path("/media/kirrog/Expansion/prunning_results_analysis_data")


def pos2childs_forming(positions_list_data):
    pos2childs = defaultdict(set)
    for i in range(len(positions_list_data)):
        for j in range(len(positions_list_data)):
            if i == j:
                continue
            l_pos = positions_list_data[i]
            r_pos = positions_list_data[j]
            if (sum([(y - x == 1) for x, y in zip(l_pos, r_pos)]) == 1 and
                    sum([(y == x) for x, y in zip(l_pos, r_pos)]) == 3):
                pos2childs[data_keys_list[i]].add(data_keys_list[j])
    return pos2childs


for dataset_path in prunning_stats_path_dir.glob("*"):
    dataset_name = str(dataset_path.name)[:-16]
    if dataset_name == "breast":
        continue
    output_path_exp_dir = output_path / dataset_name
    for prunning_exp_dir in tqdm(list(dataset_path.glob("*")), desc=f"Processing: {dataset_name}"):
        regularization_method = str(prunning_exp_dir.name).split("__")[-2:]
        epoch2stats = defaultdict(list)
        output_path_exp_dir_hyperparameters = output_path_exp_dir / prunning_exp_dir.name
        output_path_exp_dir_hyperparameters.mkdir(exist_ok=True, parents=True)
        for prunning_stats_elem_path in list(prunning_exp_dir.glob("*.json")):
            init_ep, inner_reg, weights_reg, _ = list(str(prunning_stats_elem_path.name).split("__"))
            _, _, ep_num, _, acc_val = list(init_ep.split("_"))
            inner_reg = inner_reg[10:]
            weights_reg = weights_reg[11:]

            if (f"{regularization_method[0]}___"
                f"{regularization_method[1]}___"
                f"{inner_reg}___"
                f"{weights_reg}") not in experiment_hyperparameters2compare_list:
                continue

            with open(prunning_stats_elem_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                v[1]["name"] = k
            data_keys_list = list(data.keys())
            data_keys_list.append("0_0_0_0")
            positions_list = [[int(x) // d for x, d in zip(x.split("_"), [1, 1, 8, 8])] for x in data_keys_list]

            pos2childs = pos2childs_forming(positions_list)

            ordered_stats = []
            neighbors_stats = []
            cur_str = "0_0_0_0"
            steps_features_values = list()
            deleted_features = {0: [], 1: [], 2: [], 3: []}
            while True:
                if cur_str not in pos2childs:
                    break

                childs_list = pos2childs[cur_str]
                max_childs = 0
                max_str = ""
                neigbours_current = []
                best_stats = None
                parent_nums = list(cur_str.split("_"))
                for child in childs_list:
                    childs_nums = list(child.split("_"))

                    difference_position = 0

                    for k in range(4):
                        if parent_nums[k] != childs_nums[k]:
                            difference_position = k

                    if difference_position > 1:
                        elems2del = 8
                    else:
                        elems2del = 1

                    step_acc, stats = data[child]
                    all_features_ = stats["all_features"]
                    all_features_values = [x[1] for x in all_features_]
                    features_ordered_list = list(sorted(all_features_values.copy()))

                    elems2del_pos_list = []

                    for feature_elem_value in features_ordered_list[:elems2del]:
                        counter = 0
                        elems_pos = all_features_values.index(feature_elem_value)
                        elems2del_pos_list.append(elems_pos)
                        # for i in range(len(features_ordered_list)):
                        #     if all_features_values[i] == feature_elem_value:
                        #         elems2del_pos_list.append(i)
                        # counter += 1
                        # if counter >= 2:
                        #     print(f"Double found: Exp: {exp_name} "
                        #           f"prune: {prunning_stats_elem_path.name}: {feature_elem_value}")
                    stats["elems2del_pos"] = elems2del_pos_list
                    stats["difference_position"] = difference_position
                    stats["elems2del_number"] = elems2del

                    childs_amount = len(pos2childs[child])

                    if all_features_[-1][0] != (len(all_features_) - 1):
                        print(f"Exp: {dataset_name} cut: {all_features_[-1][0]}/{len(all_features_)}")
                    if min(all_features_values) < 0.0:
                        print(f"Exp: {dataset_name} "
                              f"prune: {prunning_stats_elem_path.name}: Min: {min(all_features_values)}")

                    if childs_amount > max_childs:
                        if best_stats is not None:
                            neigbours_current.append(best_stats)
                        max_childs = childs_amount
                        best_stats = stats
                        max_str = child
                    else:
                        neigbours_current.append(stats)

                ordered_stats.append(best_stats)
                neighbors_stats.append(neigbours_current)
                cur_str = max_str

                steps_features_values_dict = dict()
                to_process = neigbours_current.copy()
                if best_stats is not None:
                    to_process.append(best_stats)
                for elem_stats in to_process:
                    all_features_ = elem_stats["all_features"]
                    all_features_copy = [x[1] for x in sorted(all_features_.copy(), key=lambda x: x[0])]
                    difference_position_ = elem_stats["difference_position"]
                    deleted_features_positions_lists = deleted_features[difference_position_]
                    for deleted_poses_i in range(len(deleted_features_positions_lists)):
                        deleted_poses = sorted(deleted_features_positions_lists[-(deleted_poses_i + 1)])
                        for i in range(len(deleted_poses)):
                            pos = deleted_poses[-(i + 1)]
                            all_features_copy.insert(pos, 0.0)
                    if len(all_features_copy) != 512 and len(all_features_copy) != 64:
                        print(f"Wrong length: {dataset_name}  {prunning_exp_dir.name}  {ep_num}  {inner_reg}  "
                              f"{weights_reg}  {cur_str} : {len(all_features_copy)}")
                        for i in range(512 - len(all_features_copy)):
                            all_features_copy.append(0.0)
                    steps_features_values_dict[elem_stats["difference_position"]] = all_features_copy

                steps_features_values.append((cur_str, steps_features_values_dict))
                if best_stats is not None:
                    elems2del_pos_ = best_stats["elems2del_pos"]
                    if len(elems2del_pos_) not in [1, 8]:
                        print(f"Double found: Exp: {dataset_name} "
                              f"prune: {prunning_stats_elem_path.name}: {elems2del_pos_}")
                    deleted_features[best_stats["difference_position"]].append(elems2del_pos_)

            ordered_len = len(ordered_stats) - 1
            neighbours_len = sum([len(x) for x in neighbors_stats])
            data_len = len(data)
            if ordered_len + neighbours_len != data_len:
                print(f"{ordered_len} + {neighbours_len} = {data_len} : {prunning_stats_elem_path}")

            block_lists = defaultdict(list)
            for cur_str_step, feature_values in steps_features_values:
                for pos, features in feature_values.items():
                    arr = np.array(features)
                    min_arr = np.min(arr)
                    max_arr = np.max(arr)
                    arr = arr - min_arr
                    arr /= max_arr
                    block_lists[pos].append(arr)

            block_stats_steps_comparison = dict()
            for pos, features in block_lists.items():
                block_comparison = []
                for i in range(1, len(features)):
                    prev_elem = features[i - 1]
                    next_elem = features[i]
                    mse = np.sum(np.abs(next_elem - prev_elem))
                    corr_coef = np.corrcoef(next_elem, prev_elem)[0, 1]
                    block_comparison.append((mse, corr_coef))
                block_stats_steps_comparison[pos] = block_comparison

            epoch2stats[int(ep_num)].append(
                (
                    acc_val,
                    inner_reg,
                    weights_reg,
                    block_stats_steps_comparison,
                    block_lists,
                    deleted_features,
                    steps_features_values
                )
            )
        for ep_num, ep_stats in epoch2stats.items():
            stats_prunning_comparison = dict()
            for i in range(len(ep_stats)):
                for j in range(i + 1, len(ep_stats)):
                    block_lists_l = ep_stats[i][4]
                    block_lists_r = ep_stats[j][4]
                    if len(block_lists_l) < 4:
                        print(f"Missed block features: Exp: {dataset_name} "
                              f"{ep_stats[i][0]}:{ep_stats[i][1]}:{ep_stats[i][2]}:{len(block_lists_l)}")
                        continue
                    if len(block_lists_r) < 4:
                        print(f"Missed block features: Exp: {dataset_name} "
                              f"{ep_stats[j][0]}:{ep_stats[j][1]}:{ep_stats[j][2]}:{len(block_lists_r)}")
                        continue
                    steps_num_min = min(
                        [len(block_lists_l[i]) for i in range(4)] +
                        [len(block_lists_r[i]) for i in range(4)]
                    )
                    step_compare_list = []
                    for k in range(steps_num_min):
                        step_compare_dict = dict()
                        for l in range(4):
                            arr_l = block_lists_l[l][k]
                            arr_r = block_lists_r[l][k]
                            mse = np.sum(np.abs(arr_l - arr_r))
                            corr_coef = np.corrcoef(arr_l, arr_r)[0, 1]
                            step_compare_dict[l] = (mse, corr_coef)
                            step_compare_list.append(step_compare_dict)
                            if len(step_compare_list) == 0:
                                continue
                            stats_prunning_comparison[f"{i}_{j}"] = step_compare_list
            ep_stats_new = []
            for exp in ep_stats:
                new_block_lists = dict()
                for k, v in exp[4].items():
                    new_block_lists[k] = [list(arr) for arr in v]
                new_exp = list(exp)
                new_exp[4] = new_block_lists
                ep_stats_new.append(new_exp)
            exp_output_path = output_path_exp_dir_hyperparameters / f"init_{ep_num}"
            exp_output_path.mkdir(exist_ok=True, parents=True)
            with open(exp_output_path / "stats.json", "w", encoding="utf-8") as f:
                json.dump(ep_stats_new, f, ensure_ascii=False)
            with open(exp_output_path / "compare.json", "w", encoding="utf-8") as f:
                json.dump(stats_prunning_comparison, f, ensure_ascii=False)
