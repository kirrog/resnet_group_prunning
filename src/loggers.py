import logging
import sys
from pathlib import Path


def create_logger(name: str, use_file_logs_writting: bool = True, std_out_or_err: bool = True):
    cstm_logger = logging.getLogger(name)
    cstm_logger.setLevel(logging.INFO)
    if use_file_logs_writting:
        path2logs = Path(f"./logs/")
        path2logs.mkdir(exist_ok=True, parents=True)
        handler = logging.FileHandler(str(path2logs / f"{name}.log"), mode='w')
        formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        cstm_logger.addHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout if std_out_or_err else sys.stderr)
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    cstm_logger.addHandler(handler)

    return logging.Logger(name)
