
#%%
from torch.utils.tensorboard import SummaryWriter

import os
import yaml
from pathlib import Path
import torch
import logging
from dataclasses import dataclass

import zarr

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@dataclass
class Config:
    labels: list
    yaml_file: str
    log_dir: str
    voxel_size: tuple
    l_rate: float
    batch_size: int
    starter_checkpoint: str
    checkpoint_classes: list

def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Config(
        labels=config["run"]["labels"],
        yaml_file=config["paths"]["yaml_file"],
        log_dir=config["paths"]["log_dir"],
        voxel_size=tuple([config["run"]["voxel_size"] for _ in range(3)]),
        l_rate=config["run"].get("l_rate", 0.5e-5),
        batch_size=config["run"].get("batch_size", 14),
        starter_checkpoint=config["checkpoint"]["path"],
        checkpoint_classes=config["checkpoint"]["classes"],
    )


def get_checkpoints(setup_path: Path):
    return list(setup_path.glob("model_checkpoint_*"))

                


def add_scores_to_tb(checkpoint_path: Path,datasets, output_path: Path, labels, thresholds = [0.4,0.5,0.6]):
    for dataset, ds_info in datasets["datasets"].items():
        for crop in ds_info["val"]:
            for label in labels:
                prediction_path = output_path/f"{checkpoint_path.name}.zarr"/crop/label

                z_pred = zarr.open(prediction_path, mode='r')
                if not "scores" in z_pred.attrs:
                    raise ValueError(f"No scores found in {prediction_path}, run score_checkpoint first")
                scores = z_pred.attrs["scores"]
                print(f"Scores for {checkpoint_path.name} on {dataset} {crop} {label}: {scores}")
                return scores
                


def validate_setup(setup_path: Path):
    if not isinstance(setup_path, Path):
        setup_path = Path(setup_path)

    config = load_config(setup_path/"config.yaml")
    labels = config.labels
    yaml_file = config.yaml_file

    checkpoints = get_checkpoints(setup_path)
    with open(setup_path/yaml_file, "r") as data_yaml:
        datasets = yaml.safe_load(data_yaml)


    output_path = setup_path / "validation"

    for checkpoint in checkpoints:
        checkpoint_path = setup_path / checkpoint
        return add_scores_to_tb(checkpoint_path, datasets, output_path, labels)

scores = validate_setup(setup_path=Path("/groups/cellmap/cellmap/zouinkhim/exp_cerebellum/runs/setup_0"))
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         logger.warning("Usage: python file.py /path/to/setup")
#         sys.exit(1)
#     setup_path = Path(sys.argv[1])
#     validate_setup(setup_path)

# %%
len(scores["balanced_accuracy_0.5"])
# %%
