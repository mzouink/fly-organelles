import logging
import math
import os

import corditea
import gunpowder as gp
import numpy as np
import torch

from fly_organelles.data import CellMapCropSource, ExtractMask
from fly_organelles.model import MaskedMultiLabelBCEwithLogits, WeightedMSELoss
from fly_organelles.utils import ShiftNorm, Binarize

logger = logging.getLogger("__name__")


def sigmoidify(arr):
    return torch.nn.functional.sigmoid(torch.tensor(arr)).numpy()


def make_affinities_data_pipeline(
    labels: list[str],
    datasets: dict,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    max_out_request: gp.Coordinate,
    displacement_sigma: gp.Coordinate,
    batch_size: int = 5,
    affinities_map = [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],]
):
    raw = gp.ArrayKey("RAW")
    
    label_keys = {}
    affinities_keys = {}
    for label in labels:
        label_keys[label] = gp.ArrayKey(label.upper())
        affinities_keys[label] = gp.ArrayKey('AFF_'+label.upper())
    
    
    srcs = []
    probs = []
    factors = {
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 2**16 - 1,
        np.dtype("int16"): 2**15 - 1  # 32767
    }
    for dataset, ds_info in datasets["datasets"].items():
        for crop in ds_info["crops"]:
            # for crop in crops.split(","):
            src = CellMapCropSource(
                ds_info["crops"][crop],
                ds_info["raw"],
                label_keys,
                raw,
                sampling,
                base_padding=pad_width_out,
                max_request=max_out_request,
            )
            src_pipe = src
            if src.needs_downsampling:
                src_pipe += corditea.AverageDownSample(raw, sampling)
            probs.append(src.get_size() / len(ds_info["crops"]))
            logging.debug(f"Padding {crop} with {src.padding}")
            for label_key, aff_key in zip(label_keys.values(), affinities_keys.values()):
                # src_pipe += Binarize(label_key)
                
                src_pipe += gp.AsType(label_key, "float32")
                src_pipe += gp.AddAffinities(
                    affinity_neighborhood=affinities_map,
                    labels=label_key,
                    affinities=aff_key,
                    dtype=np.float32)
                src_pipe += gp.Pad(label_key, src.padding, value=255.0)
                src_pipe += gp.Pad(aff_key, src.padding, value=255.0)
            # factor = factors[src.specs[raw].dtype]
            # src_pipe += gp.Normalize(raw, factor=1.0 / factor)
            # src_pipe += gp.Normalize(raw, factor=1.0)
            minc, maxc = ds_info["contrast"]
            src_pipe+= gp.AsType(raw, "float32")
            src_pipe+= ShiftNorm(raw, minc, maxc)
            # src_pipe += gp.IntensityScaleShift(raw, scale=1/(maxc - minc), shift=-(minc/(maxc-minc)) )

            # src_pipe += gp.IntensityScaleShift(raw, scale=(maxc - minc) / factor, shift=minc / factor)
            src_pipe += gp.Pad(raw, None, value=0)
            src_pipe += gp.RandomLocation()
            srcs.append(src_pipe)

    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    pipeline += gp.IntensityAugment(raw, 0.75, 1.5, -0.15, 0.15)
    pipeline += corditea.GammaAugment([raw], 0.75, 4 / 3.0)
    pipeline += gp.SimpleAugment()
    pipeline += corditea.ElasticAugment(
        control_point_spacing=gp.Coordinate((25, 25, 25)),
        control_point_displacement_sigma=displacement_sigma,
        rotation_interval=(0, math.pi / 2.0),
        subsample=8,
        uniform_3d_rotation=True,
        augmentation_probability=0.6,
    )
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += corditea.GaussianNoiseAugment(raw, var_range=(0, 0.01), noise_prob=0.5)
    # pipeline += gp.Unsqueeze(list(affinities_keys.values()))
    pipeline += corditea.Concatenate(list(affinities_keys.values()), gp.ArrayKey("LABELS"))
    # pipeline += gp.BalanceLabels(
    #     gp.ArrayKey("LABELS"),
    #     gp.ArrayKey("MASK"))
    pipeline += ExtractMask(gp.ArrayKey("LABELS"), gp.ArrayKey("MASK"))
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)
    pipeline += gp.AsType(gp.ArrayKey("LABELS"), "float32")

    return pipeline


def make_data_pipeline(
    labels: list[str],
    datasets: dict,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    max_out_request: gp.Coordinate,
    displacement_sigma: gp.Coordinate,
    batch_size: int = 5,
):
    raw = gp.ArrayKey("RAW")
    label_keys = {}
    for label in labels:
        label_keys[label] = gp.ArrayKey(label.upper())
    srcs = []
    probs = []
    factors = {
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 2**16 - 1,
        np.dtype("int16"): 2**15 - 1  # 32767
    }
    for dataset, ds_info in datasets["datasets"].items():
        for crop in ds_info["crops"]:
            # for crop in crops.split(","):
            src = CellMapCropSource(
                ds_info["crops"][crop],
                ds_info["raw"],
                label_keys,
                raw,
                sampling,
                base_padding=pad_width_out,
                max_request=max_out_request,
            )
            src_pipe = src
            if src.needs_downsampling:
                src_pipe += corditea.AverageDownSample(raw, sampling)
            probs.append(src.get_size() / len(ds_info["crops"]))
            logging.debug(f"Padding {crop} with {src.padding}")
            for label_key in label_keys.values():
                src_pipe += Binarize(label_key)
                src_pipe += gp.Pad(label_key, src.padding, value=255)
                src_pipe += gp.AsType(label_key, "float32")
            factor = factors[src.specs[raw].dtype]
            # src_pipe += gp.Normalize(raw, factor=1.0 / factor)
            # src_pipe += gp.Normalize(raw, factor=1.0)
            minc, maxc = ds_info["contrast"]
            src_pipe+= gp.AsType(raw, "float32")
            src_pipe+= ShiftNorm(raw, minc, maxc)
            # src_pipe += gp.IntensityScaleShift(raw, scale=1/(maxc - minc), shift=-(minc/(maxc-minc)) )

            # src_pipe += gp.IntensityScaleShift(raw, scale=(maxc - minc) / factor, shift=minc / factor)
            src_pipe += gp.Pad(raw, None, value=0)
            src_pipe += gp.RandomLocation()
            srcs.append(src_pipe)

    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    pipeline += gp.IntensityAugment(raw, 0.75, 1.5, -0.15, 0.15)
    pipeline += corditea.GammaAugment([raw], 0.75, 4 / 3.0)
    pipeline += gp.SimpleAugment()
    pipeline += corditea.ElasticAugment(
        control_point_spacing=gp.Coordinate((25, 25, 25)),
        control_point_displacement_sigma=displacement_sigma,
        rotation_interval=(0, math.pi / 2.0),
        subsample=8,
        uniform_3d_rotation=True,
        augmentation_probability=0.6,
    )
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += corditea.GaussianNoiseAugment(raw, var_range=(0, 0.01), noise_prob=0.5)
    pipeline += gp.Unsqueeze(list(label_keys.values()))
    pipeline += corditea.Concatenate(list(label_keys.values()), gp.ArrayKey("LABELS"))
    pipeline += ExtractMask(gp.ArrayKey("LABELS"), gp.ArrayKey("MASK"))
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)
    pipeline += gp.AsType(gp.ArrayKey("LABELS"), "float32")

    return pipeline


def make_train_pipeline(
    model,
    labels: list[str],
    label_weights: list[float],
    datasets: dict,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    max_out_request: gp.Coordinate,
    displacement_sigma: gp.Coordinate,
    input_size: gp.Coordinate,
    output_size: gp.Coordinate,
    batch_size: int = 5,
    l_rate: float = 0.5e-4,
    log_dir: str = "logs",
    affinities = False,
    affinities_map = None,
    foreground_factor = 2,
):
    if affinities:
        pipeline = make_affinities_data_pipeline(
            labels,
            datasets,
            pad_width_out,
            sampling,
            max_out_request,
            displacement_sigma,
            batch_size,
            affinities_map= affinities_map,
        )
        loss_fn = WeightedMSELoss(foreground_factor=foreground_factor)
    else:
        pipeline = make_data_pipeline(
            labels,
            datasets,
            pad_width_out,
            sampling,
            max_out_request,
            displacement_sigma,
            batch_size,
        )
        loss_fn = MaskedMultiLabelBCEwithLogits(label_weights)
    pipeline += gp.PreCache(48, 12)

    
    
    pipeline += gp.torch.Train(
    model=model,
    loss=loss_fn,
    optimizer=torch.optim.Adam(lr=l_rate, params=model.parameters()),
    inputs={"raw": gp.ArrayKey("RAW")},
    loss_inputs={"output": gp.ArrayKey("OUTPUT"), "target": gp.ArrayKey("LABELS"), "mask": gp.ArrayKey("MASK")},
    outputs={0: gp.ArrayKey("OUTPUT")},
    device="cuda",
    log_every=20,
    log_dir=log_dir,
    save_every=2000,
)


    
    pipeline += corditea.LambdaFilter(sigmoidify, gp.ArrayKey("OUTPUT"), gp.ArrayKey("NORM_OUTPUT"))
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(gp.ArrayKey("DUMMY"), input_size, voxel_size=gp.Coordinate(sampling))
    snapshot_request.add(gp.ArrayKey("NORM_OUTPUT"), output_size, voxel_size=gp.Coordinate(sampling))
    del snapshot_request[gp.ArrayKey("DUMMY")]
    pipeline += gp.Snapshot(
        {
            gp.ArrayKey("LABELS"): "labels",
            gp.ArrayKey("RAW"): "raw",
            gp.ArrayKey("MASK"): "mask",
            gp.ArrayKey("OUTPUT"): "output",
            gp.ArrayKey("NORM_OUTPUT"): "norm_output",
        },
        output_filename="{iteration:08d}.zarr",
        every=500,
        additional_request=snapshot_request,
    )
    return pipeline
