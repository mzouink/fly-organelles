import torch
import numpy as np
import gunpowder as gp
from pathlib import Path
from fly_organelles.config import get_model
from fly_organelles.utils import ShiftNorm


def predict_checkpoint_scan(
    model,
    checkpoint_path: Path,
    datasets,
    output_path: Path,
    labels,
    voxel_size,
    device,
    input_shape,
    output_shape,
    overwrite=False,
    activation_function=None,
    is_lsd=False,
    affinities_map=None,
    lsd_sigma=None,
    scores_dict: dict = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if activation_function is not None:
        model = torch.nn.Sequential(model, activation_function)
    model.to(device)
    model.eval()
    input_size = input_shape * gp.Coordinate(voxel_size)
    output_size = output_shape * gp.Coordinate(voxel_size)
    pad_width_out = output_size / 2.0
    checkpoint_name = checkpoint_path.name
    for dataset, ds_info in datasets["datasets"].items():
        if not "val" in ds_info:
            continue
        for crop in ds_info["val"]:
            for label in labels:
                raw_key = gp.ArrayKey("RAW")
                label_key = gp.ArrayKey(f"LABEL_{label}")
                prediction = gp.ArrayKey("PREDICTION")
                label_keys = {label: label_key}
                src = gp.ArraySource(
                    ds_info["val"][crop],
                    ds_info["raw"],
                    label_keys,
                    raw_key,
                    list(voxel_size),
                    base_padding=pad_width_out,
                    max_request=output_size,
                )
                pred_specs = src.specs[label_key]
                pipeline = src
                minc, maxc = ds_info["contrast"]
                pipeline += gp.AsType(raw_key, "float32")
                pipeline += ShiftNorm(raw_key, minc, maxc)
                pipeline += gp.Pad(raw_key, gp.Coordinate((None,) * len(voxel_size)))
                pipeline += gp.Unsqueeze([raw_key])
                pipeline += gp.Unsqueeze([raw_key])
                pipeline += gp.torch.Predict(
                    model=model,
                    inputs={"input": raw_key},
                    outputs={0: prediction},
                    array_specs={
                        prediction: gp.ArraySpec(roi=pred_specs.roi, voxel_size=pred_specs.voxel_size, dtype=np.float32)
                    },
                    spawn_subprocess=True,
                    device=str(device),
                )
                pipeline += gp.Squeeze([raw_key, prediction])
                pipeline += gp.ZarrWrite(
                    {
                        prediction: label + "/s0",
                    },
                    output_path / f"{checkpoint_path.name}.zarr",
                    crop,
                    dataset_dtypes={prediction: np.float32},
                )
                ref_request = gp.BatchRequest()
                ref_request.add(raw_key, input_size)
                ref_request.add(prediction, output_size)
                pipeline += gp.Scan(ref_request)
                try:
                    with gp.build(pipeline):
                        pipeline.request_batch(gp.BatchRequest())
                except Exception as e:
                    print(f"Error occurred while validating {checkpoint_path.name} on {dataset} {crop} {label}: {e}")
