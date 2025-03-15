import funlib.learn.torch
import torch

def load_eval_model(num_labels, checkpoint_path):
    model_backbone = StandardUnet(num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self, foreground_factor=2):
        super(WeightedMSELoss, self).__init__()
        self.foreground_factor = foreground_factor

    def forward(self, output, target, mask):

        weights = torch.ones_like(target)
        weights[target > 0] = self.foreground_factor

        scaled = (mask * weights * (output - target) ** 2)

        if len(torch.nonzero(scaled)) != 0:

            masked = torch.masked_select(scaled, torch.gt(mask, 0))
            loss = torch.mean(masked)

        else:
            loss = torch.mean(scaled)

        return loss
    
class MaskedMultiLabelBCEwithLogits(torch.nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight, spatial_dims=3):
        pos_weight = torch.Tensor(pos_weight)[(...,) + (None,) * spatial_dims]
        self.loss_fn = super().__init__(reduction="none", pos_weight=pos_weight)
        self.spatial_dims = spatial_dims

    def forward(self, output, target, mask):
        bce = torch.sum(super().forward(output, target) * mask)
        bce /= torch.sum(mask)
        return bce


# 154->148                                          28 -> 22
#         74-> 68                         20 -> 14
#                 34->28          16 -> 10
#                        14-> 8


# 154 -> 22
# 162 -> 30
# 170 -> 38
# 178 -> 46
# 186 -> 54
# 194 -> 62
class StandardUnet(torch.nn.Module):
    def __init__(
        self,
        out_channels,
        num_fmaps=16,
        fmap_inc_factor=6,
        downsample_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
    ):
        super().__init__()
        if downsample_factors is None:
            downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
        if kernel_size_down is None:
            kernel_size_level = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            kernel_size_down = [
                kernel_size_level,
            ] * (len(downsample_factors) + 1)
        if kernel_size_up is None:
            kernel_size_level = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            kernel_size_level = [
                kernel_size_level,
            ] * len(downsample_factors)

        self.unet_backbone = funlib.learn.torch.models.UNet(
            in_channels=1,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            constant_upsample=True,
        )

        self.final_conv = torch.nn.Conv3d(num_fmaps, out_channels, (1, 1, 1), padding="valid")

    def forward(self, raw):
        x = self.unet_backbone(raw)
        return self.final_conv(x)
