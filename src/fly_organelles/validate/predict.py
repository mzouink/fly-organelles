#%%
import warnings

# Suppress only UserWarning and FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
org = "mito"
raw_path = "/nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8"
gt_path = f"/nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/labels/groundtruth/crop522/"
output_path = "/groups/cellmap/cellmap/zouinkhim/exp_cerebellum/runs/setup_0/val"

checkpoint_path = "/groups/cellmap/cellmap/zouinkhim/exp_cerebellum/runs/setup_0/model_checkpoint_20000"


import torch
from fly_organelles.model import StandardUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StandardUnet(1)
model.eval()

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)


model = model.to(device)
#%%
import numpy as np
import gunpowder as gp
from fly_organelles.data import CellMapCropSource

resolution = 16
voxel_size = (resolution, resolution, resolution)

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




raw = gp.ArrayKey("RAW")
all_labels_key = gp.ArrayKey("LABELS")
prediction = gp.ArrayKey("PREDICTION")


src = CellMapCropSource(
    gt_path,
    raw_path,
    {"mito": all_labels_key},
    raw,
    list(voxel_size),
    base_padding=pad_width_out, 
    max_request=max_out_request,
)

#%%
pred_specs = src.specs[all_labels_key]

#%%

pipeline = src
pipeline+= gp.AsType(raw, "float32")
# raw: (c, d, h, w)
pipeline += gp.Pad(raw, gp.Coordinate((None,) * len(voxel_size)))
# raw: (c, d, h, w)
pipeline += gp.Unsqueeze([raw])

pipeline += gp.Unsqueeze([raw])
# raw: (1, c, d, h, w)


# predict
pipeline += gp.torch.Predict(
    model=model,
    inputs={"raw": raw},
    outputs={0: prediction},
    array_specs={
        prediction: gp.ArraySpec(
            roi=pred_specs.roi, voxel_size=pred_specs.voxel_size, dtype=np.float32
        )
    },
    spawn_subprocess=False,
    device="cpu",
)
# raw: (1, c, d, h, w)
# prediction: (1, [c,] d, h, w)

# prepare writing
pipeline += gp.Squeeze([raw, prediction])
pipeline += gp.Squeeze([raw, prediction])
# raw: (c, d, h, w)
# prediction: (c, d, h, w)
# raw: (c, d, h, w)
# prediction: (c, d, h, w)

# write to zarr
pipeline += gp.ZarrWrite(
    {prediction: "mito_pred"},
    f"{output_path}/pred.zarr",
    "mito_pred",
    dataset_dtypes={prediction: np.float32},
)


# create reference batch request
ref_request = gp.BatchRequest()
ref_request.add(raw, input_size)
ref_request.add(prediction, output_size)
pipeline += gp.Scan(ref_request)

# build pipeline and predict in complete output ROI

with gp.build(pipeline):
    pipeline.request_batch(gp.BatchRequest())



# %%

help(model.forward)
# %%
f"{output_path}/pred.zarr/mito_pred/mito_pred",


# %%
import zarr
z = zarr.open(f"{output_path}/pred.zarr/mito_pred/mito_pred", mode="r")
# %%
z.shape
# %%
z2 = zarr.open(f"/nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/labels/groundtruth/crop522/mito/s2","r")
z2.shape
# %%
