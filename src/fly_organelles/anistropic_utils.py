from pydantic_ome_ngff.v04.multiscale import (
    Dataset,
    MultiscaleGroupAttrs,
    MultiscaleMetadata,
)
from statistics import mode
from pydantic_ome_ngff.v04.axis import Axis
from typing import Optional
import itertools
import numpy as np
from pydantic_ome_ngff.v04.transform import VectorScale, VectorTranslation

def ax_dict_to_list(ax_dict, axes_order):
    return [ax_dict[ax] for ax in axes_order]

def get_downsampling_factors(samplings):
    sorted_keys = sorted(samplings.keys(), key=lambda x: min(samplings[x]))
    factors = []
    for sc1, sc2 in itertools.pairwise(sorted_keys):
        factor = []
        for i in range(len(samplings[sc1])):
            factor.append(samplings[sc2][i] / samplings[sc1][i])
        factors.append(factor)
    return factors

def infer_nominal_transform(scale: [float], offset: [float]) -> tuple[dict[str, int], dict[str, int]]:
    # print(scale, offset)
    if len(scale) != len(offset):
        msg = f"Length of offset {len(offset)} does not match length of scale {len(scale)}."
        raise ValueError(msg)
    if len(set(scale)) == len(scale):
        msg = "Scales along all axes are unique, cannot infer nominal transform."
        raise ValueError(msg)
    # get dominant scale, i.e. the one that's represented more than once with np unique
    nominal_scale_val = mode(scale)
    if int(nominal_scale_val) != nominal_scale_val:
        msg = f"Dominant scale {nominal_scale_val} is not an integer, cannot infer nominal transform."
        raise ValueError(msg)
    nominal_scale_val = int(nominal_scale_val)
    nominal_scale = [nominal_scale_val,]*len(scale)
    nominal_offset = []
    for ax, off in enumerate(offset):
        pix_off = off / scale[ax]
        # assert np.isclose(int(pix_off * 2), pix_off * 2, 1e-4), f"{pix_off}"
        nominal_offset.append(int(round(pix_off * nominal_scale_val)))
    return nominal_scale, nominal_offset

def change_multiscale_name(attrs, name):
    if len(attrs["multiscales"]) > 1:
        msg = f"Multiscales attribute in {attrs.name} already contains several entries. Selection is not implemented for renaming."
        raise NotImplementedError(msg)
    attrs["multiscales"][0]["name"] = name
    return attrs

def get_axes_object(zarr_grp, multiscale_name: Optional[str] = None) -> list[Axis]:
    if multiscale_name is None:
        index = 0
    else:
        for index, multiscale in enumerate(zarr_grp.attrs["multiscales"]):
            if multiscale.get("name") == multiscale_name:
                break
        else:
            # raise an error if no matching multiscale found
            msg = f"Multiscale with name '{multiscale_name}' not found in Zarr group at {zarr_grp.store.path}"
            raise KeyError(msg)
    msattrs = MultiscaleGroupAttrs(multiscales=zarr_grp.attrs["multiscales"])
    return msattrs.multiscales[index].axes

def generate_standard_multiscale(dataset_paths, axes, base_resolution, base_offset, factors, name="nominal"):
    scale = np.array(base_resolution)
    offset = np.array(base_offset)
    transforms = [(VectorScale(scale=tuple(scale)), VectorTranslation(translation=tuple(offset)))]
    for factor in factors:
        offset = offset + (factor - np.ones_like(factor)) * scale / 2.0
        scale = scale * factor
        transforms.append(
            (
                VectorScale(scale=tuple(scale)),
                VectorTranslation(translation=tuple(offset)),
            )
        )
    datasets = tuple(
        Dataset(path=dataset_path, coordinateTransformations=transform)
        for dataset_path, transform in zip(dataset_paths, transforms)
    )
    return MultiscaleMetadata(name=name, axes=axes, type=None, datasets=datasets)

# %%
