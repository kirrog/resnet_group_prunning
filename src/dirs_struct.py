import datetime
from pathlib import Path
from typing import Tuple


class DirsStruct:
    root_path_models: Path = Path("/media/kirrog/data/data/fqwb_data/models")
    root_path_stats: Path = Path("/media/kirrog/data/data/fqwb_data/stats")

    def get_stats__and_model_save_path(self, experiment_name: str) -> Tuple[Path, Path]:
        now_datetime = datetime.datetime.now()

        model_experiment_path = self.root_path_models / f"{now_datetime.strftime('%Y_%m_%d__%H_%M')}___{experiment_name}"
        stats_experiment_path = self.root_path_stats / f"{now_datetime.strftime('%Y_%m_%d__%H_%M')}___{experiment_name}"

        model_experiment_path.mkdir(parents=True, exist_ok=True)
        stats_experiment_path.mkdir(parents=True, exist_ok=True)

        return model_experiment_path, stats_experiment_path
