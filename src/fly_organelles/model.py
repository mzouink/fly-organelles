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


import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedAffinitiesLoss(nn.Module):
    def __init__(
        self,
        affinities_weight: float = 1.0,
        lsds_weight: float = 1.0,
        num_affinities_channels: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.affinities_weight = affinities_weight
        self.lsds_weight = lsds_weight
        self.num_affinities = num_affinities_channels
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bs = output.shape[0]

        # --- Affinities branch ---
        out_aff  = output[:, :self.num_affinities]
        tgt_aff  = target[:, :self.num_affinities]
        mask_aff = mask[:, :self.num_affinities].float()
        C_aff    = out_aff.shape[1]

        m_flat = mask_aff.view(bs, C_aff, -1)
        t_flat = (tgt_aff > 0.0).float().view(bs, C_aff, -1)
        total_valid = m_flat.sum(dim=(0,2))
        pos = (t_flat * m_flat).sum(dim=(0,2))
        neg = total_valid - pos

        pos_w = (neg / (total_valid + self.eps))
        neg_w = (pos / (total_valid + self.eps))
        shape = [1, C_aff] + [1] * (out_aff.ndim-2)
        pos_w = pos_w.view(*shape)
        neg_w = neg_w.view(*shape)

        weight_aff = mask_aff * ( (tgt_aff>0.0).float()*pos_w + (tgt_aff==0.0).float()*neg_w )
        bce    = F.binary_cross_entropy_with_logits(out_aff, tgt_aff, reduction="none")
        w_bce  = bce * weight_aff

        denom = weight_aff.sum()
        if denom.detach().item() < self.eps:
            loss_aff = bce.mean()
        else:
            loss_aff = w_bce.sum() / (denom + self.eps) / C_aff

        # --- LSDS branch ---
        out_lsds  = output[:, self.num_affinities:]
        tgt_lsds  = target[:, self.num_affinities:]
        mask_lsds = mask[:, self.num_affinities:].float()
        C_lsds    = out_lsds.shape[1]

        m_flat = mask_lsds.view(bs, C_lsds, -1)
        t_flat = (tgt_lsds > 0.0).float().view(bs, C_lsds, -1)
        total_valid = m_flat.sum(dim=(0,2))
        pos = (t_flat * m_flat).sum(dim=(0,2))
        neg = total_valid - pos

        pos_w = (neg / (total_valid + self.eps))
        neg_w = (pos / (total_valid + self.eps))
        shape = [1, C_lsds] + [1] * (out_lsds.ndim-2)
        pos_w = pos_w.view(*shape)
        neg_w = neg_w.view(*shape)

        weight_lsds = mask_lsds * ( (tgt_lsds>0.0).float()*pos_w + (tgt_lsds==0.0).float()*neg_w )
        mse    = F.mse_loss(out_lsds, tgt_lsds, reduction="none")
        w_mse  = mse * weight_lsds

        denom = weight_lsds.sum()
        if denom.detach().item() < self.eps:
            loss_lsds = mse.mean()
        else:
            loss_lsds = w_mse.sum() / (denom + self.eps) / C_lsds

        return self.affinities_weight * loss_aff + self.lsds_weight * loss_lsds
    
class AffinitiesLoss(nn.Module):
    def __init__(self, lsds_weight: float = 1.0, affinities_weight: float = 1.0, nb_affinities: int = 3):
        super().__init__()
        self.lsds_weight = lsds_weight
        self.affinities_weight = affinities_weight
        self.nb_affinities = nb_affinities

    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        # Validate channel and mask dimensions
        if output.shape != target.shape:
            raise ValueError(f"Output and target must have the same shape, got {output.shape} vs {target.shape}")
        if mask.shape != output.shape:
            raise ValueError(f"Mask must match output shape, got {mask.shape} vs {output.shape}")
        if output.shape[1] < self.nb_affinities:
            raise ValueError(f"Expected at least {self.nb_affinities} affinity channels, got {output.shape[1]}")

        # Split predictions, targets, and mask into affinities and LSDS parts
        out_aff = output[:, :self.nb_affinities]
        tgt_aff = target[:, :self.nb_affinities]
        mask_aff = mask[:, :self.nb_affinities].float()

        out_lsds = output[:, self.nb_affinities:]
        tgt_lsds = target[:, self.nb_affinities:]
        mask_lsds = mask[:, self.nb_affinities:].float()

        bce = torch.nn.BCEWithLogitsLoss(reduction="none")(out_aff, tgt_aff) * mask_aff
        print(bce.mean())

        loss_lsds = torch.nn.MSELoss(reduction="none")(torch.nn.Sigmoid()(out_lsds), tgt_lsds) * mask_lsds
        print(loss_lsds.mean())

        return self.affinities_weight * bce.mean() + self.lsds_weight * loss_lsds.mean()


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self, eps: float = 1e-6, foreground_factor: float = 1.0):
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
