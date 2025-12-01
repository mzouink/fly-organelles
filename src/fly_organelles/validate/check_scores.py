# %%
p1 = "/nrs/cellmap/data/jrc_mus-cerebellum-3/jrc_mus-cerebellum-3.zarr/recon-1/labels/groundtruth/crop1055/lyso/s0"
p2 = "/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_lyso_mouse_distance_16nm/validation/model_checkpoint_354000.zarr/crop1055/lyso/s0"
import zarr

z1 = zarr.open(p1, mode="r")[:]
z2 = zarr.open(p2, mode="r")[0]
print(z1.shape, z2.shape)
# %%
import numpy as np


# f1 score
def f1_score_new(y_true, y_pred, threshold=0.5):
    print(y_true.shape, y_pred.shape)
    y_pred = (y_pred >= threshold).astype(np.uint8)
    y_true = (y_true >= threshold).astype(np.uint8)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


from fly_organelles.validate.score import f1_score

# %%
score = f1_score_new(z1, z2)
# %%
import torch

tensor1 = torch.from_numpy(z1)
tensor2 = torch.from_numpy(z2)
score_torch = f1_score(tensor2, tensor1)
# %%
print("F1 score numpy:", score)
print("F1 score torch:", score_torch)
# %%
