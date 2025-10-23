import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
import json
import torch
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
    affinities_map: list | None
    lsd_sigma: float | None

    @property
    def total_labels(self) -> int:
        if not self.is_lsd:
            return len(self.labels)
        else:
            affinities_count = len(self.affinities_map)
            return len(self.labels) * (10 + affinities_count)

    def check(self):
        if self.is_lsd and self.lsd_sigma is None:
            raise ValueError("lsd_sigma must be specified if lsd is True")
        if not self.is_lsd and self.lsd_sigma is not None:
            logger.warning("lsd_sigma is specified but lsd is False, ignoring lsd_sigma")
        if self.starter_checkpoint is None:
            logger.warning("No starter checkpoint specified, training will start from scratch")
        if self.model_type not in ["standard_unet", "isolated_unet"]:
            raise ValueError(f"model_type must be 'standard_unet' or 'isolated_unet', got {self.model_type}")
        if self.model_type == "isolated_unet" and self.model_json_config is None:
            raise ValueError("model_json_config must be specified if model_type is 'isolated_unet'")
        if self.affinities_map is None and self.is_lsd:
            raise ValueError("affinities_map must be specified if lsd is True")


def get_model(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.model_type == "isolated_unet":
        activation_function = None
        if self.model_json_config is None:
            raise ValueError("model_json_config must be provided for isolated_unet model type")
        from fly_organelles.isolated_unet import UNetLSD, load_checkpoint_from_path

        with open(self.model_json_config, "r") as f:
            model_config = json.load(f)
        model = UNetLSD(**model_config)
        if self.starter_checkpoint is not None:
            load_checkpoint_from_path(model, self.starter_checkpoint, map_location=device)
        else:
            logger.warning("Didn't load any pretrained weights for isolated_unet")
    elif self.model_type == "standard_unet":
        from fly_organelles.model import StandardUnet

        model = StandardUnet(self.total_labels)
        activation_function = torch.nn.Sigmoid()

        logger.warning("Didn't load any pretrained weights for standard_unet")

    else:
        raise ValueError(f"Unknown model type: {self.model_type}")

    model.eval()
    model = model.to(device)
    return model, activation_function


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    val_yaml = config["paths"].get("val_yaml", None)
    if val_yaml is None:
        logger.error("val_yaml must be specified in the config file under paths. will use yaml_file instead.")
        val_yaml = config["paths"]["yaml_file"]
    config = Config(
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
        affinities_map=config["run"].get("affinities_map", None),
        lsd_sigma=config["run"].get("lsd_sigma", None),
    )
    config.check()
    return config
