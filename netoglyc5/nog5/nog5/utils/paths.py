from pathlib import Path
from typing import Tuple
import datetime

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
RUN_DIR = "runs"


def ensure_exists(p: Path) -> Path:
    """ Helper to ensure a directory exists. """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def arch_path(config: dict) -> Path:
    """ Construct a path based on the name of a configuration file eg. 'saved/EfficientNet' """
    p = Path(config["save_dir"]) / config["name"]
    return ensure_exists(p)


def log_path(config: dict) -> Path:
    p = arch_path(config) / LOG_DIR
    return ensure_exists(p)


def trainer_paths(config: dict) -> Tuple[Path, Path]:
    """ Returns the paths to save checkpoints and tensorboard runs. eg. """
    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    arch_datetime = arch_path(config) / start_time
    return (
        ensure_exists(arch_datetime / CHECKPOINT_DIR),
        ensure_exists(arch_datetime / RUN_DIR),
    )


def optim_results_path(results_path: str, config: dict) -> Path:
    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    p = Path(results_path) / config["name"] / start_time
    return ensure_exists(p)
