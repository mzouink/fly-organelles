# %%
import argparse
import os

import dask
import fibsem_tools as fst
import neuroglancer
import neuroglancer.cli
import numpy as np
from pathlib import Path
from fly_organelles.validate.validate_run import load_config
from fly_organelles.utils import find_target_scale

config_base_name = "config.yaml"


def create_lv(path, volume_type="segmentation", array_name="raw", offset=None, voxel_size=None):
    z_arr = fst.access(os.path.join(path, array_name), mode="a")
    if offset is not None:
        z_arr.attrs["offset"] = offset
    if voxel_size is not None:
        z_arr.attrs["voxel_size"] = voxel_size
    g_arr = z_arr
    print(z_arr)

    arr = fst.io.zarr.core.to_dask(z_arr)
    if volume_type == "segmentation":
        arr = arr.astype("uint8")

    dim_names = ["z", "y", "x"]
    dim_units = ["nm", "nm", "nm"]
    dim_scales = g_arr.attrs["voxel_size"]
    voxel_offset = np.array(g_arr.attrs["offset"]) / np.array(g_arr.attrs["voxel_size"])
    for dim in range(arr.ndim)[::-1]:
        if arr.shape[dim] == 1:
            arr = dask.array.squeeze(arr, axis=dim)
            dim_names.pop(dim)
            dim_units.pop(dim)
            dim_scales.pop(dim)
            voxel_offset.pop(dim)
    dims = neuroglancer.CoordinateSpace(names=dim_names, units=dim_units, scales=dim_scales)
    return neuroglancer.LocalVolume(arr, dimensions=dims, volume_type=volume_type, voxel_offset=voxel_offset)


def create_lv_stacked(
    checkpoints_path, crop, volume_type="segmentation", array_name="raw", offset=None, voxel_size=None
):
    dask_arrs = []
    for checkpoint in checkpoints_path:
        z_arr = fst.access(os.path.join(checkpoint, crop, array_name, "s0"), mode="a")
        if offset is not None:
            z_arr.attrs["offset"] = offset
        if voxel_size is not None:
            z_arr.attrs["voxel_size"] = voxel_size
        dask_arr = fst.io.zarr.core.to_dask(z_arr)
        if volume_type == "segmentation":
            dask_arr = dask_arr.astype("uint8")
        dask_arrs.append(dask_arr)
    dask_arrs = dask.array.stack(dask_arrs, axis=-1)

    # Determine dimension names based on the array shape
    # Assume the original array is [c, z, y, x] or [z, y, x]
    if dask_arrs.ndim == 5:  # [c, z, y, x, t]
        dim_names = ["c^", "z", "y", "x", "t"]
        dim_units = ["", "nm", "nm", "nm", "s"]
        dim_scales = [1, *z_arr.attrs["voxel_size"], 1]
        voxel_offset = [0, *(np.array(z_arr.attrs["offset"]) / np.array(z_arr.attrs["voxel_size"])), 0]
    else:  # [z, y, x, t]
        dim_names = ["z", "y", "x", "t"]
        dim_units = ["nm", "nm", "nm", "s"]
        dim_scales = [*z_arr.attrs["voxel_size"], 1]
        voxel_offset = [*(np.array(z_arr.attrs["offset"]) / np.array(z_arr.attrs["voxel_size"])), 0]
    for dim in range(dask_arrs.ndim)[::-1]:
        if dask_arrs.shape[dim] == 1:
            dask_arrs = dask.array.squeeze(dask_arrs, axis=dim)
            dim_names.pop(dim)
            dim_units.pop(dim)
            dim_scales.pop(dim)
            voxel_offset.pop(dim)
    dims = neuroglancer.CoordinateSpace(names=dim_names, units=dim_units, scales=dim_scales)
    return neuroglancer.LocalVolume(dask_arrs, dimensions=dims, volume_type=volume_type, voxel_offset=voxel_offset)


def set_layers(state, checkpoints_path, crop, organelle, raw_path, gt_path, resolution):
    layers = {
        "raw": "image",
        "output": "image",
        "norm_output": "image",
        "multi_labels": "image",
        "labels": "segmentation",
        "mask": "segmentation",
    }

    gt_scale, gt_offset, gt_shape = find_target_scale(fst.read(gt_path), resolution)
    print("GT scale:", gt_scale)
    print("GT offset:", gt_offset)
    print("GT shape:", gt_shape)

    mounted_raw_path = raw_path.replace("/nrs/cellmap/", "zarr://https://cellmap-vm1.int.janelia.org/nrs/")
    mounted_gt_path = gt_path.replace("/nrs/cellmap/", "zarr://https://cellmap-vm1.int.janelia.org/nrs/")

    state.layers["raw"] = neuroglancer.ImageLayer(source=mounted_raw_path)
    state.layers["labels"] = neuroglancer.SegmentationLayer(source=mounted_gt_path)
    state.layers["predictions"] = neuroglancer.ImageLayer(
        source=create_lv_stacked(
            checkpoints_path, crop, volume_type="image", array_name=organelle, offset=gt_offset, voxel_size=resolution
        )
    )

    # if add_time:
    #     lv_func = create_lv_stacked
    # else:
    #     lv_func = create_lv
    # for layer_name, layer_type in layers.items():
    #     if layer_type == "image":
    #         state.layers[layer_name] = neuroglancer.ImageLayer(
    #             source=lv_func(snapshot_path, volume_type=layer_type, array_name=layer_name)
    #         )
    #     else:
    #         state.layers[layer_name] = neuroglancer.SegmentationLayer(
    #             source=lv_func(snapshot_path, volume_type=layer_type, array_name=layer_name)
    #         )


def get_raw(datasets, crop):
    for d, v in datasets["datasets"].items():
        if "val" not in v:
            continue
        raw_path = v["raw"]
        if crop in v["val"]:
            return raw_path, v["val"][crop]
    return None, None


def main(run_path):
    config_path = os.path.join(run_path, config_base_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {run_path}")
    config = load_config(Path(config_path))
    data_path = config.val_yaml
    if "/" not in data_path:
        data_path = os.path.join(run_path, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found in {data_path}")
    validation_path = os.path.join(run_path, "validation")
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Validation folder not found in {run_path}")
    checkpoints_path = [
        os.path.join(validation_path, d)
        for d in os.listdir(validation_path)
        if d.startswith("model_checkpoint_") and d.endswith(".zarr")
    ]
    crops = [d for d in os.listdir(checkpoints_path[0]) if os.path.isdir(os.path.join(checkpoints_path[0], d))]
    organelles = [
        d
        for d in os.listdir(os.path.join(checkpoints_path[0], crops[0]))
        if os.path.isdir(os.path.join(checkpoints_path[0], crops[0], d))
    ]

    neuroglancer.set_server_bind_address("0.0.0.0")
    # neuroglancer.cli.add_server_arguments(ap)

    # neuroglancer.cli.handle_server_arguments(args)
    import yaml

    with open(data_path, 'r') as f:
        datasets = yaml.safe_load(f)
    # print(datasets["datasets"])

    resolution = config.voxel_size

    viewer = neuroglancer.Viewer()

    for crop in crops:
        for organelle in organelles:
            raw_path, gt_path = get_raw(datasets, crop)
            if raw_path is None:
                raise ValueError(f"Crop {crop} not found in datasets")
            print(f"Adding {crop}/{organelle}")
            org_gt_path = os.path.join(gt_path, organelle)

            print(raw_path, org_gt_path)

            with viewer.txn() as s:
                # Clear existing layers
                for layer in list(s.layers):
                    del s.layers[str(layer)]

                # Add new layers
                set_layers(s, checkpoints_path, crop, organelle, raw_path, org_gt_path, resolution)

            print(viewer)
            yield viewer


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("run_path", type=str)

#     args = ap.parse_args()
#     main(args.run_path)

# Use the generator
for viewer in main("/groups/cellmap/cellmap/zouinkhim/exp_salivary/runs/setup_42"):
    print("New link generated:")
    print(viewer)
    input("Press Enter to generate the next link...")
    # You can break after the first one or continue to iterate through all
    # break


# %%
