# %%
import os
import json
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
from fly_organelles.utils import ShiftNorm, generate_4d_scale_attrs
from fly_organelles.validate.score import f1_score
from fly_organelles.utils import find_target_scale
from fly_organelles.config import load_config, get_model
import sys
from fly_organelles.validate.predict_v2 import predict_checkpoint_v2

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_scores_yaml(setup_path: Path) -> dict:
    """Load existing scores from scores.yaml if it exists."""
    scores_file = setup_path / "scores.yaml"
    if scores_file.exists():
        with open(scores_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_scores_yaml(setup_path: Path, scores_dict: dict):
    """Save scores to scores.yaml, merging with existing scores to support concurrent jobs.
    Only updates scores if they are higher than existing scores (based on F1 score)."""
    scores_file = setup_path / "scores.yaml"

    # Load existing scores from file (in case another job updated it)
    existing_scores = {}
    if scores_file.exists():
        try:
            with open(scores_file, "r") as f:
                existing_scores = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error loading existing scores: {e}. Will overwrite.")

    # Deep merge: update existing_scores with new scores_dict, keeping higher scores
    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge update dict into base dict.
        For score dictionaries (containing 'f1', 'accuracy', etc.), only update if new F1 is higher."""
        for key, value in update.items():
            if key in base:
                # Check if both are dictionaries
                if isinstance(base[key], dict) and isinstance(value, dict):
                    # Check if this is a scores dict (has 'f1' key)
                    if 'f1' in value and 'f1' in base[key]:
                        # This is a scores dictionary - only update if new F1 is higher
                        if value['f1'] > base[key]['f1']:
                            logger.warning(f"Updating scores for {key}: F1 {base[key]['f1']:.4f} -> {value['f1']:.4f}")
                            base[key] = value
                        else:
                            logger.warning(
                                f"Keeping existing scores for {key}: F1 {base[key]['f1']:.4f} >= {value['f1']:.4f}"
                            )
                    else:
                        # Not a scores dict, recurse deeper
                        deep_merge(base[key], value)
                else:
                    # Not both dicts, overwrite
                    base[key] = value
            else:
                # Key doesn't exist, add it
                base[key] = value
        return base

    merged_scores = deep_merge(existing_scores, scores_dict)

    # Save merged scores
    with open(scores_file, "w") as f:
        yaml.dump(merged_scores, f, default_flow_style=False, sort_keys=False)


def set_metadata(input_ds: Path, out_ds: Path):
    z_from = zarr.open(input_ds, mode='r')
    attrs_s0 = z_from.attrs["multiscales"][0]["datasets"][0]
    scale = attrs_s0["coordinateTransformations"][0]["scale"]
    translation = attrs_s0["coordinateTransformations"][1]["translation"]
    attrs_new = generate_4d_scale_attrs(scale, translation)
    z_to = zarr.open(out_ds, mode='a')
    z_to.attrs["multiscales"] = attrs_new["multiscales"]


def get_checkpoints(setup_path: Path):
    return list(setup_path.glob("model_checkpoint_*"))


def get_missing_checkpoints(setup_path: Path) -> list[Path]:
    """Get checkpoints that haven't been scored yet.

    Args:
        setup_path: Path to the setup directory containing checkpoints and scores.yaml

    Returns:
        List of checkpoint paths that are missing scores in scores.yaml
    """
    if not isinstance(setup_path, Path):
        setup_path = Path(setup_path)

    # Get all available checkpoints
    all_checkpoints = get_checkpoints(setup_path)

    if not all_checkpoints:
        logger.warning(f"No checkpoints found in {setup_path}")
        return []

    # Load existing scores
    scores_dict = load_scores_yaml(setup_path)

    # Get checkpoint names that have been scored
    scored_checkpoint_names = set(scores_dict.keys())

    # Find checkpoints that haven't been scored
    missing_checkpoints = []
    for checkpoint in all_checkpoints:
        checkpoint_name = checkpoint.name
        if checkpoint_name not in scored_checkpoint_names:
            missing_checkpoints.append(checkpoint)
            logger.info(f"Missing scores for: {checkpoint_name}")
        else:
            # Even if checkpoint is in scores.yaml, check if it has complete scores
            # by verifying it has at least one score entry
            checkpoint_scores = scores_dict[checkpoint_name]
            if not checkpoint_scores or not any(isinstance(v, dict) for v in checkpoint_scores.values()):
                missing_checkpoints.append(checkpoint)
                logger.info(f"Incomplete scores for: {checkpoint_name}")

    # Sort by iteration number
    missing_checkpoints = sorted(missing_checkpoints, key=lambda x: int(x.name.split("_")[-1]))

    logger.warning(
        f"Found {len(missing_checkpoints)} checkpoints without complete scores out of {len(all_checkpoints)} total"
    )

    return missing_checkpoints


def score_checkpoint(
    checkpoint_path: Path,
    datasets,
    output_path: Path,
    labels,
    voxel_size,
    device,
    tb_writer: SummaryWriter = None,
    is_lsd=False,
    affinities_map=None,
    setup_path: Path = None,
    scores_dict: dict = None,
    overwrite: bool = True,
):
    val_loss = torch.nn.BCEWithLogitsLoss().to(device)
    checkpoint_name = checkpoint_path.name

    # Initialize checkpoint entry in scores_dict if needed
    if scores_dict is not None and checkpoint_name not in scores_dict:
        scores_dict[checkpoint_name] = {}

    for dataset, ds_info in datasets["datasets"].items():
        if not "val" in ds_info:
            logger.warning(f"Skipping {dataset} as it has no validation set.")
            continue
        for crop in ds_info["val"]:
            try:
                for label in labels:
                    # Check if scores already exist for this checkpoint/dataset/crop/label
                    if scores_dict is not None and not overwrite:
                        if checkpoint_name in scores_dict:
                            if dataset in scores_dict[checkpoint_name]:
                                if crop in scores_dict[checkpoint_name][dataset]:
                                    if label in scores_dict[checkpoint_name][dataset][crop]:
                                        # Check if F1 score is < 0.5 (likely buggy pre-sigmoid scores)
                                        existing_scores = scores_dict[checkpoint_name][dataset][crop][label]
                                        f1_score_val = existing_scores.get('f1', 0)
                                        if f1_score_val < 0.5:
                                            logger.warning(
                                                f"Re-scoring {checkpoint_name} on {dataset}/{crop}/{label} - F1={f1_score_val:.4f} (likely buggy score)"
                                            )
                                        else:
                                            logger.warning(
                                                f"Skipping scoring for {checkpoint_name} on {dataset}/{crop}/{label} - F1={f1_score_val:.4f} already good"
                                            )
                                            continue

                    label_grp = fst.read(Path(ds_info["val"][crop]) / label)
                    label_scale, _, _ = find_target_scale(label_grp, list(voxel_size))
                    gt_path = Path(ds_info["val"][crop]) / label / label_scale
                    prediction_path = output_path / f"{checkpoint_path.name}.zarr" / crop / label / "s0"
                    z_gt = zarr.open(gt_path, mode='r')
                    z_pred = zarr.open(prediction_path, mode='r')

                    # Handle LSD predictions - extract relevant channels
                    if is_lsd:
                        # For LSD, the prediction has shape (num_affinities + 10, d, h, w)
                        # We need to convert ground truth to affinities as well for proper comparison
                        pred_array = z_pred[:]
                        if affinities_map is not None and len(affinities_map) > 0:
                            # Get affinity predictions (first num_affs channels)
                            num_affs = len(affinities_map)
                            pred_ts = torch.from_numpy(pred_array[:num_affs]).float().to(device)

                            # Convert ground truth to affinities using the same method as training
                            from fly_organelles.lsds.lite.affs import get_affs as get_affs_lite

                            gt_binary = np.array(z_gt[:], dtype=np.uint8)
                            gt_affs = get_affs_lite(
                                gt_binary, neighborhood=affinities_map, dist="equality-no-bg", pad=True
                            ).astype(np.float32)
                            label_ts = torch.from_numpy(gt_affs).float().to(device)
                        else:
                            logger.warning(f"No affinities_map provided for LSD scoring, using first channel")
                            pred_ts = torch.from_numpy(pred_array[0]).float().to(device)
                            gt_array = np.array(z_gt[:], dtype=np.float32)
                            label_ts = (torch.from_numpy(gt_array) > 0).float().to(device)
                    else:
                        pred_ts = torch.from_numpy(z_pred[:]).float().to(device)
                        gt_array = np.array(z_gt[:], dtype=np.float32)
                        label_ts = (torch.from_numpy(gt_array) > 0).long().to(device)

                    # Apply sigmoid activation to convert logits to probabilities
                    # Model outputs are logits, need to apply sigmoid before thresholding
                    # TODO this is a fix for fly models only
                    pred_ts = torch.sigmoid(pred_ts)

                    scores = {}
                    threshold = 0.5
                    pred_thresh = (pred_ts > threshold).long()

                    if is_lsd and affinities_map is not None and len(affinities_map) > 0:
                        # For affinities, label_ts is already float (affinities values)
                        # We need to binarize it for comparison
                        label_thresh = (label_ts > threshold).long()
                        # Calculate metrics per affinity channel and average
                        f1_scores = []
                        accuracies = []
                        for ch in range(pred_thresh.shape[0]):
                            f1_scores.append(f1_score(label_thresh[ch], pred_thresh[ch]))
                            correct = (pred_thresh[ch] == label_thresh[ch]).sum()
                            total = torch.numel(label_thresh[ch])
                            accuracies.append((correct / total).cpu().numpy().item())

                        scores["f1"] = np.mean(f1_scores)
                        scores["accuracy"] = np.mean(accuracies)
                        scores["val_loss"] = val_loss(pred_thresh.float(), label_thresh.float()).cpu().numpy().item()
                    else:
                        # Regular binary segmentation
                        scores["f1"] = f1_score(label_ts, pred_thresh)
                        scores["val_loss"] = val_loss(pred_thresh.float(), label_ts.float()).cpu().numpy().item()
                        correct = (pred_thresh == label_ts).sum()
                        total = torch.numel(label_ts)
                        accuracy = correct / total
                        scores["accuracy"] = accuracy.cpu().numpy().item()

                    if tb_writer is not None:
                        tb_writer.add_scalar(
                            f"val/{dataset}_{crop}_{label}/f1",
                            scores["f1"],
                            global_step=int(checkpoint_path.name.split("_")[-1]),
                        )
                        tb_writer.add_scalar(
                            f"val/{dataset}_{crop}_{label}/val_loss",
                            scores["val_loss"],
                            global_step=int(checkpoint_path.name.split("_")[-1]),
                        )
                        tb_writer.add_scalar(
                            f"val/{dataset}_{crop}_{label}/accuracy",
                            scores["accuracy"],
                            global_step=int(checkpoint_path.name.split("_")[-1]),
                        )
                    logger.warning(f"Scores for {checkpoint_path.name} on {dataset} {crop} {label}: {scores}")

                    # Save scores to the parent zarr group (not the s0 array)
                    z_pred_group = zarr.open(output_path / f"{checkpoint_path.name}.zarr" / crop / label, mode='a')
                    z_pred_group.attrs["scores"] = scores

                    # Save scores to the scores_dict
                    if scores_dict is not None:
                        if dataset not in scores_dict[checkpoint_name]:
                            scores_dict[checkpoint_name][dataset] = {}
                        if crop not in scores_dict[checkpoint_name][dataset]:
                            scores_dict[checkpoint_name][dataset][crop] = {}
                        # Convert numpy types to Python native types for YAML serialization
                        clean_scores = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in scores.items()
                        }
                        scores_dict[checkpoint_name][dataset][crop][label] = clean_scores
            except Exception as e:
                logger.error(f"Error scoring {checkpoint_path.name} on {dataset} {crop} {label}: {e}")
                raise e


def update_tensorboard(setup_path: Path):
    """Update TensorBoard with scores from scores.yaml.

    This function reads the scores.yaml file and adds all scores to TensorBoard.
    Useful for backfilling TensorBoard when scores exist but weren't written to TB.

    Args:
        setup_path: Path to the setup directory containing scores.yaml and config.yaml
    """
    if not isinstance(setup_path, Path):
        setup_path = Path(setup_path)

    # Load config to get log_dir
    config = load_config(setup_path / "config.yaml")
    log_dir = config.log_dir

    # Load scores from scores.yaml
    scores_dict = load_scores_yaml(setup_path)

    if not scores_dict:
        logger.warning(f"No scores found in {setup_path / 'scores.yaml'}")
        return

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Iterate through all scores and add to TensorBoard
    for checkpoint_name, checkpoint_scores in scores_dict.items():
        # Extract iteration number from checkpoint name (e.g., model_checkpoint_10000 -> 10000)
        try:
            iteration = int(checkpoint_name.split("_")[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract iteration from checkpoint name: {checkpoint_name}")
            continue

        for dataset, dataset_scores in checkpoint_scores.items():
            for crop, crop_scores in dataset_scores.items():
                for label, scores in crop_scores.items():
                    if isinstance(scores, dict):
                        # Write each metric to TensorBoard
                        if "f1" in scores:
                            writer.add_scalar(
                                f"val/{dataset}_{crop}_{label}/f1",
                                scores["f1"],
                                global_step=iteration,
                            )
                        if "val_loss" in scores:
                            writer.add_scalar(
                                f"val/{dataset}_{crop}_{label}/val_loss",
                                scores["val_loss"],
                                global_step=iteration,
                            )
                        if "accuracy" in scores:
                            writer.add_scalar(
                                f"val/{dataset}_{crop}_{label}/accuracy",
                                scores["accuracy"],
                                global_step=iteration,
                            )

                        logger.info(f"Added scores for {checkpoint_name} on {dataset}/{crop}/{label} to TensorBoard")

    # Flush and close writer
    writer.flush()
    writer.close()
    logger.warning(f"Successfully updated TensorBoard with scores from {setup_path / 'scores.yaml'}")


def validate_setup(setup_path: Path, checkpoint_iteration: int | None = None, force_gpu: bool = True):
    if not isinstance(setup_path, Path):
        setup_path = Path(setup_path)

    # Force GPU check
    if force_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU is required (force_gpu=True) but CUDA is not available!")

    config = load_config(setup_path / "config.yaml")
    labels = config.labels
    yaml_file = config.val_yaml
    voxel_size = config.voxel_size
    voxel_size = gp.Coordinate(voxel_size)
    log_dir = config.log_dir
    writer = SummaryWriter(log_dir=log_dir)
    input_shape = config.input_shape
    output_shape = config.output_shape

    checkpoints = get_checkpoints(setup_path)

    # Filter for specific checkpoint if provided
    if checkpoint_iteration is not None:
        checkpoints = [cp for cp in checkpoints if int(cp.name.split("_")[-1]) == checkpoint_iteration]
        if not checkpoints:
            raise ValueError(f"Checkpoint with iteration {checkpoint_iteration} not found in {setup_path}")
        logger.warning(f"Validating specific checkpoint: model_checkpoint_{checkpoint_iteration}")
    with open(setup_path / yaml_file, "r") as data_yaml:
        datasets = yaml.safe_load(data_yaml)

    total_labels = config.total_labels
    is_lsd = config.is_lsd
    affinities_map = config.affinities_map
    lsd_sigma = config.lsd_sigma
    model, activation_function = get_model(config)

    output_path = setup_path / "validation"
    if not output_path.exists():
        os.makedirs(output_path)

    # Load existing scores
    scores_dict = load_scores_yaml(setup_path)

    # Sort checkpoints by iteration number (highest to lowest)
    checkpoints = sorted(
        checkpoints, key=lambda x: int(x.name.split("_")[-1]), reverse=True  # Start with highest iteration
    )

    for checkpoint in checkpoints:
        checkpoint_path = setup_path / checkpoint
        predict_checkpoint_v2(
            model,
            checkpoint_path,
            datasets,
            output_path,
            labels,
            voxel_size,
            device,
            input_shape=input_shape,
            output_shape=output_shape,
            overwrite=False,  # Don't re-predict if already have scores
            activation_function=activation_function,
            # is_lsd=is_lsd,
            # affinities_map=affinities_map,
            # lsd_sigma=lsd_sigma,
            scores_dict=scores_dict,
        )
        score_checkpoint(
            checkpoint_path,
            datasets,
            output_path,
            labels,
            voxel_size,
            device,
            tb_writer=writer,
            is_lsd=is_lsd,
            affinities_map=affinities_map,
            setup_path=setup_path,
            scores_dict=scores_dict,
            overwrite=False,  # Don't re-score if already scored
        )

        # Save scores after each checkpoint
        save_scores_yaml(setup_path, scores_dict)

        # Flush TensorBoard writer to ensure scores are written to disk
        writer.flush()

    # Close the TensorBoard writer when all checkpoints are done
    writer.close()


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        logger.warning(
            "Usage: python file.py /path/to/setup [checkpoint_iteration|--list-missing|--update-tensorboard]"
        )
        logger.warning(
            "  checkpoint_iteration: optional iteration number (e.g., 10000) to validate a specific checkpoint"
        )
        logger.warning("  --list-missing: list checkpoints that haven't been scored yet")
        logger.warning("  --update-tensorboard: update TensorBoard with scores from scores.yaml")
        sys.exit(1)

    setup_path = Path(sys.argv[1])

    # Check if user wants to list missing checkpoints
    if len(sys.argv) == 3 and sys.argv[2] == "--list-missing":
        missing = get_missing_checkpoints(setup_path)
        if missing:
            print(f"\nFound {len(missing)} checkpoint(s) without scores:")
            for cp in missing:
                iteration = int(cp.name.split("_")[-1])
                print(f"  - {cp.name} (iteration {iteration})")
        else:
            print("\nAll checkpoints have been scored!")
        sys.exit(0)

    # Check if user wants to update TensorBoard from scores.yaml
    if len(sys.argv) == 3 and sys.argv[2] == "--update-tensorboard":
        update_tensorboard(setup_path)
        sys.exit(0)

    checkpoint_iteration = int(sys.argv[2]) if len(sys.argv) == 3 else None
    validate_setup(setup_path, checkpoint_iteration)


if __name__ == "__main__":
    main()
# validate_setup(Path("/groups/cellmap/cellmap/zouinkhim/exp_pancreas/runs/setup_43"))
