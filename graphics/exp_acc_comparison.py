from collections import defaultdict
from pathlib import Path
from pprint import pprint

models_root_dir = Path("/media/kirrog/data/data/fqwb_data/models")

init_stats = dict()
init_versions = defaultdict(dict)

for model_exp in models_root_dir.glob("*"):
    model_exp_features = str(model_exp.name).split("___")
    base_name = model_exp_features[1]

    best_acc = 0.0
    best_ep = 0
    max_ep = 0
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
        if max_ep < int_ep:
            max_ep = int_ep

    name_features = list(base_name.split("__"))
    if len(name_features) == 1:
        init_stats[base_name] = {"acc": best_acc, "best": best_ep, "last": max_ep}
    else:
        init_version = name_features[1]
        reg_func = name_features[2]
        coefs = name_features[3]
        if reg_func not in init_versions[init_version]:
            init_versions[init_version][reg_func] = []
        init_versions[init_version][reg_func].append({"coefs": coefs,
                                                      "stats": {"acc": best_acc, "best": best_ep, "last": max_ep}})

print(init_stats)
print(init_versions)
stats_table = defaultdict(dict)
for init_v, v in init_versions.items():
    for reg_func, stats in v.items():
        best_acc = 0.0
        best_ep = 0
        for stat in stats:

            acc = stat["stats"]["acc"]
            ep = stat["stats"]["best"]
            if best_acc < acc:
                best_acc = acc
                best_ep = ep
        stats_table[init_v][reg_func] = f"{acc}_{ep}"

pprint(stats_table)
