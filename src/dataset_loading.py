from pathlib import Path
from typing import List

import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class DataSet:
    data_np: np.array
    data_lb: List[int]

    def __init__(self, data: np.array, labels: List[int]):
        self.data_np = data
        self.data_lb = labels


def load_dataset(paths: List[Path]) -> DataSet:
    datas = []
    labels = []
    for p in paths:
        d = unpickle(str(p))
        datas.append(d[b'data'])
        labels.extend(d[b'labels'])
    data = np.concatenate(datas, axis=0)
    return DataSet(data, labels)


def load_data(train_paths: List[Path],
              valid_paths: List[Path],
              test_paths: List[Path]) -> (DataSet, DataSet, DataSet):
    return load_dataset(train_paths), load_dataset(valid_paths), load_dataset(test_paths)


