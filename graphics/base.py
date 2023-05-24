import pickle
from pathlib import Path


def load_exp(path2metrics: Path):
    results = dict()
    for hyps in path2metrics.glob("*"):
        pickles_data = []
        for ep in list(sorted(list(hyps.glob("*")), key=lambda x: int(x.name[3:6]))):
            with open(str(ep), "rb") as f:
                data = pickle.load(f)
                pickles_data.append(data)
        results[str(hyps.name)] = pickles_data
    return results


def load_metrics(path2metrics: Path):
    orig = path2metrics / "aug_4_block"
    block = path2metrics / "aug_4_block_reg_block"
    group = path2metrics / "aug_4_block_reg_group"
    orig_metrics = load_exp(orig)
    block_metrics = load_exp(block)
    group_metrics = load_exp(group)
    return orig_metrics, block_metrics, group_metrics
