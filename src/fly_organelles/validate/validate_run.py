
#%%
import os
import yaml
from pathlib import Path
import torch
import numpy as np
import logging
from dataclasses import dataclass
import gunpowder as gp
import corditea
import zarr
import fibsem_tools as fst
from torch.utils.tensorboard import SummaryWriter
from fly_organelles.model import StandardUnet
from fly_organelles.data import CellMapCropSource
from fly_organelles.utils import ShiftNorm
from fly_organelles.validate.score import f1_score
from fly_organelles.utils import find_target_scale
import sys
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

def predict_checkpoint(model, checkpoint_path: Path,datasets, output_path: Path, labels, voxel_size,device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    
    model = torch.nn.Sequential(model, torch.nn.Sigmoid())
    model.eval()
    input_size = gp.Coordinate((178, 178, 178))
    output_size = gp.Coordinate((56, 56, 56))
    input_size = gp.Coordinate(input_size) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate(output_size) * gp.Coordinate(voxel_size)
    displacement_sigma = gp.Coordinate((24, 24, 24))
    #    max_in_request = gp.Coordinate((np.ceil(np.sqrt(sum(input_size**2))),)*len(input_size)) + displacement_sigma * 6
    max_out_request = (
    gp.Coordinate((np.ceil(np.sqrt(sum(output_size**2))),) * len(output_size)) + displacement_sigma * 6
    )
    pad_width_out = output_size / 2.0

    for dataset, ds_info in datasets["datasets"].items():
        if not "val" in ds_info:
            logger.warning(f"Skipping {dataset} as it has no validation set.")
            continue
        for crop in ds_info["val"]:
            for label in labels:
                if (output_path/f"{checkpoint_path.name}.zarr"/crop/label).exists():
                    logger.warning(f"Skipping {checkpoint_path.name} as it has already been validated. found {output_path/f'{checkpoint_path.name}.zarr'/crop/label}")
                    return
                logger.warning(f"Validating {checkpoint_path.name} on {dataset} {crop} {label}")
                raw_key = gp.ArrayKey("RAW")
                label_key = gp.ArrayKey(f"LABEL_{label}")
                prediction = gp.ArrayKey("PREDICTION")
                label_keys = {label: label_key}

                src = CellMapCropSource(
                    ds_info["val"][crop],
                    ds_info["raw"],
                    label_keys,
                    raw_key,
                    list(voxel_size),
                    base_padding=pad_width_out,
                    max_request=max_out_request,
                )

                pred_specs = src.specs[label_key]
                pipeline = src
                if src.needs_downsampling:
                    pipeline += corditea.AverageDownSample(raw_key, list(voxel_size))
                # logging.debug(f"Padding {crop} with {src.padding}")

                minc, maxc = ds_info["contrast"]
                pipeline += gp.AsType(raw_key, "float32")
                pipeline += ShiftNorm(raw_key, minc, maxc)

                pipeline += gp.Pad(raw_key, gp.Coordinate((None,) * len(voxel_size)))
                # raw: (c, d, h, w)
                pipeline += gp.Unsqueeze([raw_key])

                pipeline += gp.Unsqueeze([raw_key])
                # raw: (1, c, d, h, w)


                # predict
                pipeline += gp.torch.Predict(
                    model=model,
                    inputs={"input": raw_key},
                    outputs={0: prediction},
                    array_specs={
                        prediction: gp.ArraySpec(
                            roi=pred_specs.roi, voxel_size=pred_specs.voxel_size, dtype=np.float32
                        )
                    },
                    spawn_subprocess=False,
                    device=str(device),
                )
                # raw: (1, c, d, h, w)
                # prediction: (1, [c,] d, h, w)

                # prepare writing
                pipeline += gp.Squeeze([raw_key, prediction])
                pipeline += gp.Squeeze([raw_key, prediction])
                # raw: (c, d, h, w)
                # prediction: (c, d, h, w)
                # raw: (c, d, h, w)
                # prediction: (c, d, h, w)

                # write to zarr
                pipeline += gp.ZarrWrite(
                    {prediction: label},
                    output_path/f"{checkpoint_path.name}.zarr",
                    crop,
                    dataset_dtypes={prediction: np.float32},
                )

                # create reference batch request
                ref_request = gp.BatchRequest()
                ref_request.add(raw_key, input_size)
                ref_request.add(prediction, output_size)
                pipeline += gp.Scan(ref_request)

                # build pipeline and predict in complete output ROI

                with gp.build(pipeline):
                    pipeline.request_batch(gp.BatchRequest())
                    

                


def score_checkpoint(checkpoint_path: Path,datasets, output_path: Path, labels, voxel_size,device, activation_function =  torch.nn.Sigmoid(), thresholds = [0.5], tb_writer: SummaryWriter = None):
    val_loss = torch.nn.BCEWithLogitsLoss().to(device)
    for dataset, ds_info in datasets["datasets"].items():
        if not "val" in ds_info:
            logger.warning(f"Skipping {dataset} as it has no validation set.")
            continue
        for crop in ds_info["val"]:
            for label in labels:
                label_grp = fst.read(Path(ds_info["val"][crop]) / label)
                label_scale, _, _ = find_target_scale(label_grp, list(voxel_size))
                gt_path = Path(ds_info["val"][crop]) / label / label_scale
                prediction_path = output_path/f"{checkpoint_path.name}.zarr"/crop/label
                z_gt = zarr.open(gt_path, mode='r')
                z_pred = zarr.open(prediction_path, mode='a')
                # if "scores" in z_pred.attrs:
                #     logger.warning(f"Skipping {checkpoint_path.name} on {dataset} {crop} {label} as it has already been scored.")
                #     continue
                pred_ts = torch.from_numpy(z_pred[:]).float().to(device)
                label_ts = (torch.from_numpy(z_gt[:])>0).long().to(device)
                # if activation_function is not None:
                #     pred_ts = activation_function(pred_ts)
                scores = {}
                for threshold in thresholds:
                    pred_thresh = (pred_ts > threshold).long()
                    scores[f"f1_{threshold}"] = f1_score(label_ts, pred_thresh)
                    scores[f"val_loss_{threshold}"] = val_loss(pred_thresh.float(), label_ts.float()).cpu().numpy().item()
                    correct = (pred_thresh == label_ts).sum()
                    total = torch.numel(label_ts)
                    accuracy = correct / total
                    scores[f"accuracy_{threshold}"] = accuracy.cpu().numpy().item()
                    if tb_writer is not None:
                        tb_writer.add_scalar(f"val/{dataset}_{crop}_{label}/f1_{threshold}", scores[f"f1_{threshold}"], global_step=int(checkpoint_path.name.split("_")[-1]))
                        tb_writer.add_scalar(f"val/{dataset}_{crop}_{label}/val_loss_{threshold}", scores[f"val_loss_{threshold}"], global_step=int(checkpoint_path.name.split("_")[-1]))
                        tb_writer.add_scalar(f"val/{dataset}_{crop}_{label}/accuracy_{threshold}", scores[f"accuracy_{threshold}"], global_step=int(checkpoint_path.name.split("_")[-1]))
                logger.warning(f"Scores for {checkpoint_path.name} on {dataset} {crop} {label}: {scores}")

                z_pred.attrs["scores"] = scores
               
                


def validate_setup(setup_path: Path):
    if not isinstance(setup_path, Path):
        setup_path = Path(setup_path)

    config = load_config(setup_path/"config.yaml")
    labels = config.labels
    yaml_file = config.yaml_file
    voxel_size = config.voxel_size
    voxel_size = gp.Coordinate(voxel_size)
    log_dir = config.log_dir
    writer = SummaryWriter(log_dir=log_dir)

    checkpoints = get_checkpoints(setup_path)
    with open(setup_path/yaml_file, "r") as data_yaml:
        datasets = yaml.safe_load(data_yaml)


    model = StandardUnet(len(labels))
    model.eval()
    model = model.to(device)
    output_path = setup_path / "validation"
    if not output_path.exists():
        os.makedirs(output_path)

    for checkpoint in checkpoints:
        checkpoint_path = setup_path / checkpoint
        predict_checkpoint(model, checkpoint_path, datasets, output_path, labels, voxel_size, device)
        score_checkpoint(checkpoint_path, datasets, output_path, labels, voxel_size, device, tb_writer=writer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.warning("Usage: python file.py /path/to/setup")
        sys.exit(1)
    setup_path = Path(sys.argv[1])
    validate_setup(setup_path)
