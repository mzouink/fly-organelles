import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
import gunpowder as gp

logger = logging.getLogger(__name__)


@dataclass
class Config:
    labels: list
    yaml_file: str
    val_yaml: str
    log_dir: str
    voxel_size: tuple
    l_rate: float
    batch_size: int
    starter_checkpoint: str
    checkpoint_classes: list
    is_lsd: bool
    input_shape: gp.Coordinate
    output_shape: gp.Coordinate
    model_type: str
    model_json_config: str


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    val_yaml = config["paths"].get("val_yaml", None)
    if val_yaml is None:
        logger.error("val_yaml must be specified in the config file under paths. will use yaml_file instead.")
        val_yaml = config["paths"]["yaml_file"]
    return Config(
        labels=config["run"]["labels"],
        yaml_file=config["paths"]["yaml_file"],
        val_yaml=val_yaml,
        log_dir=config["paths"]["log_dir"],
        voxel_size=tuple([config["run"]["voxel_size"] for _ in range(3)]),
        l_rate=config["run"].get("l_rate", 0.5e-5),
        batch_size=config["run"].get("batch_size", 14),
        starter_checkpoint=config["checkpoint"].get("path", None),
        checkpoint_classes=config["checkpoint"].get("classes", []),
        model_json_config=config["checkpoint"].get("model_json_config", None),
        is_lsd=config["run"].get("lsd", False),
        input_shape=gp.Coordinate(config["checkpoint"].get("input_shape", [178, 178, 178])),
        output_shape=gp.Coordinate(config["checkpoint"].get("output_shape", [56, 56, 56])),
        model_type=config["checkpoint"].get("model_type", "standard_unet"),
    )
