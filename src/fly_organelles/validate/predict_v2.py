import torch
import numpy as np
import gunpowder as gp
from pathlib import Path
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
import fibsem_tools as fst
from fly_organelles.config import get_model
from fly_organelles.utils import find_target_scale, ShiftNorm


def predict_checkpoint_v2(
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
    scores_dict: dict = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if activation_function is not None:
        model = torch.nn.Sequential(model, activation_function)
    model.to(device)
    model.eval()
    input_size = Coordinate(input_shape) * Coordinate(voxel_size)
    output_size = Coordinate(output_shape) * Coordinate(voxel_size)
    context = (input_size - output_size) / 2

    for dataset, ds_info in datasets["datasets"].items():
        if "val" not in ds_info:
            continue
        for crop in ds_info["val"]:
            # Prepare raw data (once per crop)
            raw_grp = fst.read(ds_info["raw"])
            raw_scale, raw_offset, raw_shape = find_target_scale(raw_grp, list(voxel_size))
            raw_array_path = f"{ds_info['raw']}/{raw_scale}"
            raw_xarray = fst.read_xarray(raw_array_path)
            raw_roi = gp.Roi(Coordinate(raw_offset), Coordinate(raw_shape) * Coordinate(voxel_size))
            minc, maxc = ds_info["contrast"]
            output_path_ds = output_path / f"{checkpoint_path.name}.zarr"
            # Prepare output zarr group for all labels
            blockwise_group = str(output_path_ds)
            # For each label, process and export
            for label in labels:
                label_crop_ds_path = f"{ds_info['val'][crop]}/{label}"
                label_grp = fst.read(label_crop_ds_path)
                label_scale, label_offset, label_shape = find_target_scale(label_grp, list(voxel_size))
                label_xarray = fst.read_xarray(f"{label_crop_ds_path}/{label_scale}")
                label_offset = [int(label_xarray.coords[axis][0].data) for axis in "zyx"]
                label_voxel_size = [
                    int((label_xarray.coords[axis][1] - label_xarray.coords[axis][0]).data) for axis in "zyx"
                ]
                label_shape = list(label_xarray.shape)
                output_roi = gp.Roi(Coordinate(label_offset), Coordinate(label_shape) * Coordinate(label_voxel_size))
                spatial_shape = output_roi.shape / Coordinate(voxel_size)
                test_input = torch.zeros(1, 1, *input_shape).to(device)
                with torch.no_grad():
                    test_output = model(test_input)
                    num_channels = test_output.shape[1]
                shape_with_channels = (num_channels,) + tuple(spatial_shape)
                # Prepare prediction dataset for this label
                pred_ds_name = f"{crop}/{label}/s0"
                prediction_array = prepare_ds(
                    blockwise_group + f"/{pred_ds_name}",
                    shape=shape_with_channels,
                    offset=output_roi.offset,
                    voxel_size=Coordinate(voxel_size),
                    dtype=np.float32,
                    axis_names=["c^", "z", "y", "x"],
                    mode="w",
                )
                # Prepare raw dataset (once per crop, but per label for schema compatibility)
                raw_ds_name = f"{crop}/raw/s0"
                raw_shape = (1,) + tuple(spatial_shape)
                raw_array = prepare_ds(
                    blockwise_group + f"/{raw_ds_name}",
                    shape=raw_shape,
                    offset=output_roi.offset,
                    voxel_size=Coordinate(voxel_size),
                    dtype=np.float32,
                    axis_names=["c^", "z", "y", "x"],
                    mode="w",
                )
                block_count = 0
                total_shape = output_roi.shape
                for z_offset in range(0, int(total_shape[0]), int(output_size[0])):
                    for y_offset in range(0, int(total_shape[1]), int(output_size[1])):
                        for x_offset in range(0, int(total_shape[2]), int(output_size[2])):
                            block_offset = output_roi.offset + Coordinate([z_offset, y_offset, x_offset])
                            block_shape = Coordinate(
                                [
                                    min(output_size[0], total_shape[0] - z_offset),
                                    min(output_size[1], total_shape[1] - y_offset),
                                    min(output_size[2], total_shape[2] - x_offset),
                                ]
                            )
                            target_write_roi = gp.Roi(block_offset, block_shape)
                            target_center = target_write_roi.offset + target_write_roi.shape / 2
                            ideal_input_roi = gp.Roi(target_center - input_size / 2, input_size)
                            predicted_output_offset = ideal_input_roi.offset + context
                            actual_input_roi = ideal_input_roi.intersect(raw_roi)
                            input_offset_shift = actual_input_roi.offset - ideal_input_roi.offset
                            input_shape_diff = ideal_input_roi.shape - actual_input_roi.shape
                            z_start, y_start, x_start = actual_input_roi.get_begin()
                            z_end, y_end, x_end = actual_input_roi.get_end()
                            # Debug prints for ROI and shapes
                            print(f"Block {block_count+1}: write_roi={target_write_roi}, input_roi={actual_input_roi}")
                            print(f"  input_offset_shift={input_offset_shift}, input_shape_diff={input_shape_diff}")
                            print(f"  zyx start-end: {z_start}-{z_end}, {y_start}-{y_end}, {x_start}-{x_end}")

                            # Use same slicing logic as benchmark (no -1)
                            raw_input = raw_xarray.sel(
                                z=slice(z_start, z_end),
                                y=slice(y_start, y_end),
                                x=slice(x_start, x_end),
                            )
                            raw_input = np.asarray(raw_input)
                            # Pad if needed
                            pad_before = tuple(
                                int(max(0, -shift // vs)) for shift, vs in zip(input_offset_shift, voxel_size)
                            )
                            pad_after = tuple(int(max(0, diff // vs)) for diff, vs in zip(input_shape_diff, voxel_size))
                            if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
                                raw_input = np.pad(
                                    raw_input, tuple((pb, pa) for pb, pa in zip(pad_before, pad_after)), mode='edge'
                                )
                            print(f"  raw_input shape after pad: {raw_input.shape}")
                            raw_input = raw_input.astype(np.float32)
                            raw_input = (raw_input - minc) / (maxc - minc)
                            raw_input = raw_input * 2 - 1
                            raw_input = np.expand_dims(raw_input, 0)
                            raw_input = np.expand_dims(raw_input, 0)
                            print(f"  raw_input shape for model: {raw_input.shape}")
                            # Ensure input block is exactly input_shape
                            target_shape = tuple(input_shape)
                            current_shape = raw_input.shape[-3:]
                            pad_needed = [max(0, t - c) for t, c in zip(target_shape, current_shape)]
                            crop_needed = [max(0, c - t) for t, c in zip(target_shape, current_shape)]
                            # Pad if needed
                            if any(pad_needed):
                                pad_width = [(0, 0), (0, 0)] + [(0, p) for p in pad_needed]
                                raw_input = np.pad(raw_input, pad_width, mode='edge')
                                print(f"  padded raw_input to {raw_input.shape}")
                            # Crop if needed
                            if any(crop_needed):
                                slices = [slice(None), slice(None)] + [slice(0, t) for t in target_shape]
                                raw_input = raw_input[tuple(slices)]
                                print(f"  cropped raw_input to {raw_input.shape}")
                            # Final check
                            assert raw_input.shape[-3:] == tuple(
                                input_shape
                            ), f"Input shape mismatch: {raw_input.shape[-3:]} vs {input_shape}"
                            with torch.no_grad():
                                predictions = (
                                    model.forward(torch.from_numpy(raw_input).float().to(device))
                                    .detach()
                                    .cpu()
                                    .numpy()[0]
                                )
                            print(f"  predictions shape: {predictions.shape}")
                            predicted_shape_voxels = Coordinate(predictions.shape[1:])
                            predicted_shape = predicted_shape_voxels * Coordinate(voxel_size)
                            predicted_roi = gp.Roi(predicted_output_offset, predicted_shape)
                            actual_write_roi = target_write_roi.intersect(predicted_roi).intersect(output_roi)
                            if not actual_write_roi.empty:
                                pred_relative = actual_write_roi - predicted_roi.offset
                                pred_slices = tuple(
                                    slice(int(b // vs), int(e // vs))
                                    for b, e, vs in zip(
                                        pred_relative.get_begin(),
                                        pred_relative.get_end(),
                                        voxel_size,
                                    )
                                )
                                predictions_to_write = predictions[(slice(None),) + pred_slices]
                                prediction_array[actual_write_roi] = predictions_to_write
                                # Export raw
                                raw_relative = actual_write_roi - output_roi.offset
                                raw_slices = tuple(
                                    slice(int(b // vs), int(e // vs))
                                    for b, e, vs in zip(
                                        raw_relative.get_begin(),
                                        raw_relative.get_end(),
                                        voxel_size,
                                    )
                                )
                                raw_to_write = raw_input[0, 0][raw_slices]
                                raw_to_write = np.expand_dims(raw_to_write, 0)
                                raw_array[actual_write_roi] = raw_to_write
                            block_count += 1
                            print(f"Processed block {block_count}", end="\r")
                    print(f"\nProcessed {block_count} blocks for {label}")
