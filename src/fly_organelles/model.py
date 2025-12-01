import funlib.learn.torch
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AffinitiesLoss(nn.Module):
    def __init__(self, lsds_weight: float = 0.4, affinities_weight: float = 0.6, nb_affinities: int = 3):
        super().__init__()
        self.lsds_weight = lsds_weight
        self.affinities_weight = affinities_weight
        self.nb_affinities = nb_affinities

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # Validate channel and mask dimensions
        if output.shape != target.shape:
            raise ValueError(f"Output and target must have the same shape, got {output.shape} vs {target.shape}")
        if mask.shape != output.shape:
            raise ValueError(f"Mask must match output shape, got {mask.shape} vs {output.shape}")
        if output.shape[1] < self.nb_affinities:
            raise ValueError(f"Expected at least {self.nb_affinities} affinity channels, got {output.shape[1]}")

        # Split predictions, targets, and mask into affinities and LSDS parts
        out_aff = output[:, : self.nb_affinities]
        tgt_aff = target[:, : self.nb_affinities]
        mask_aff = mask[:, : self.nb_affinities].float()

        out_lsds = output[:, self.nb_affinities :]
        tgt_lsds = target[:, self.nb_affinities :]
        mask_lsds = mask[:, self.nb_affinities :].float()

        bce = torch.nn.BCEWithLogitsLoss(reduction="none")(out_aff, tgt_aff) * mask_aff
        # print(bce.mean())

        loss_lsds = torch.nn.MSELoss(reduction="none")(out_lsds, tgt_lsds) * mask_lsds
        # print(loss_lsds.mean())

        return self.affinities_weight * bce.mean() + self.lsds_weight * loss_lsds.mean()


class AffinitiesLossV2(nn.Module):
    def __init__(
        self,
        lsds_weight: float = 1.0,
        affinities_weight: float = 1.0,
        nb_affinities: int = 3,
        affinity_channel_weights: list[float] | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal_loss: bool = True,
        boundary_emphasis: bool = True,
        lsds_separate_weights: bool = True,
    ):
        """
        Improved loss for affinities + LSDs

        Args:
            lsds_weight: Weight for LSDS loss component
            affinities_weight: Weight for affinity loss component
            nb_affinities: Number of affinity channels
            affinity_channel_weights: Per-channel weights for affinities (e.g., lower for long-range)
            focal_alpha: Class balance factor for focal loss
            focal_gamma: Focusing parameter for focal loss (higher = more focus on hard examples)
            use_focal_loss: Use focal loss instead of BCE for affinities
            boundary_emphasis: Increase loss weight near object boundaries
            lsds_separate_weights: Use different weights for offset vs shape LSDs
        """
        super().__init__()
        self.lsds_weight = lsds_weight
        self.affinities_weight = affinities_weight
        self.nb_affinities = nb_affinities
        self.affinity_channel_weights = affinity_channel_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal_loss = use_focal_loss
        self.boundary_emphasis = boundary_emphasis
        self.lsds_separate_weights = lsds_separate_weights

    def focal_loss(self, pred_logits, target, alpha=0.25, gamma=2.0):
        """
        Focal loss for handling class imbalance
        Focuses on hard examples by down-weighting easy ones
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pred_prob = torch.sigmoid(pred_logits)

        # p_t: probability of correct class
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma

        # Alpha weighting for class balance
        alpha_t = alpha * target + (1 - alpha) * (1 - target)

        return alpha_t * focal_weight * bce_loss

    def get_boundary_mask(self, target_aff, kernel_size=3):
        """
        Create a mask that emphasizes boundaries
        Returns higher weights near object boundaries
        """
        # Use max pooling to detect edges
        ndim = target_aff.ndim - 2  # exclude batch and channel dims
        if ndim == 3:
            max_pool = F.max_pool3d(target_aff, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            min_pool = -F.max_pool3d(-target_aff, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        else:
            max_pool = F.max_pool2d(target_aff, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            min_pool = -F.max_pool2d(-target_aff, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # Boundary where there's variation in local neighborhood
        boundary = (max_pool - min_pool) > 0.1

        # Boost boundary regions (e.g., 2x weight)
        boundary_weight = torch.ones_like(target_aff)
        boundary_weight[boundary] = 2.0

        return boundary_weight

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # Validate channel and mask dimensions
        if output.shape != target.shape:
            raise ValueError(f"Output and target must have the same shape, got {output.shape} vs {target.shape}")
        if mask.shape != output.shape:
            raise ValueError(f"Mask must match output shape, got {mask.shape} vs {output.shape}")
        if output.shape[1] < self.nb_affinities:
            raise ValueError(f"Expected at least {self.nb_affinities} affinity channels, got {output.shape[1]}")

        # Split predictions, targets, and mask into affinities and LSDS parts
        out_aff = output[:, : self.nb_affinities]
        tgt_aff = target[:, : self.nb_affinities]
        mask_aff = mask[:, : self.nb_affinities].float()

        out_lsds = output[:, self.nb_affinities :]
        tgt_lsds = target[:, self.nb_affinities :]
        mask_lsds = mask[:, self.nb_affinities :].float()

        # ====== AFFINITY LOSS ======
        if self.use_focal_loss:
            # Focal loss - better for class imbalance
            aff_loss = self.focal_loss(out_aff, tgt_aff, self.focal_alpha, self.focal_gamma)
        else:
            # Standard BCE
            aff_loss = F.binary_cross_entropy_with_logits(out_aff, tgt_aff, reduction='none')

        # Apply mask
        aff_loss = aff_loss * mask_aff

        # Boundary emphasis
        if self.boundary_emphasis:
            boundary_weight = self.get_boundary_mask(tgt_aff)
            aff_loss = aff_loss * boundary_weight

        # Per-channel weighting (e.g., lower weight for long-range affinities)
        if self.affinity_channel_weights is not None:
            weights = torch.tensor(self.affinity_channel_weights, device=aff_loss.device, dtype=aff_loss.dtype)
            weights = weights.view(1, -1, *([1] * (aff_loss.ndim - 2)))
            aff_loss = aff_loss * weights

        aff_loss = aff_loss.mean()

        # ====== LSDS LOSS ======
        if self.lsds_separate_weights:
            # Different weights for different LSDS components
            # Assuming 3D: 3 offset + 3 variance + 3 pearson + 1 mass = 10 channels
            ndims = 3  # or infer from out_lsds.shape[1]
            n_offset = ndims
            n_variance = ndims
            n_pearson = ndims * (ndims - 1) // 2

            # Offset channels (first ndims) - most important, higher weight
            if out_lsds.shape[1] >= n_offset:
                loss_offset = (
                    F.mse_loss(out_lsds[:, :n_offset], tgt_lsds[:, :n_offset], reduction='none')
                    * mask_lsds[:, :n_offset]
                )
                loss_offset = loss_offset.mean()
            else:
                loss_offset = 0.0

            # Variance channels - important for shape
            if out_lsds.shape[1] >= n_offset + n_variance:
                loss_variance = (
                    F.mse_loss(
                        out_lsds[:, n_offset : n_offset + n_variance],
                        tgt_lsds[:, n_offset : n_offset + n_variance],
                        reduction='none',
                    )
                    * mask_lsds[:, n_offset : n_offset + n_variance]
                )
                loss_variance = loss_variance.mean()
            else:
                loss_variance = 0.0

            # Pearson + mass - lower weight
            if out_lsds.shape[1] > n_offset + n_variance:
                loss_shape = (
                    F.mse_loss(
                        out_lsds[:, n_offset + n_variance :], tgt_lsds[:, n_offset + n_variance :], reduction='none'
                    )
                    * mask_lsds[:, n_offset + n_variance :]
                )
                loss_shape = loss_shape.mean()
            else:
                loss_shape = 0.0

            # Weighted combination (offset most important)
            loss_lsds = 2.0 * loss_offset + 1.5 * loss_variance + 1.0 * loss_shape
        else:
            # Standard MSE across all LSDS channels
            loss_lsds = F.mse_loss(out_lsds, tgt_lsds, reduction='none') * mask_lsds
            loss_lsds = loss_lsds.mean()

        total_loss = self.affinities_weight * aff_loss + self.lsds_weight * loss_lsds

        return total_loss


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation - helps with over-splitting and false positives
    Can be used as an auxiliary loss alongside AffinitiesLoss
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, output, target, mask=None):
        output = torch.sigmoid(output)

        if mask is not None:
            output = output * mask
            target = target * mask

        intersection = (output * target).sum(dim=(2, 3, 4))
        union = output.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss with affinities, LSDs, and optional auxiliary losses
    """

    def __init__(
        self,
        nb_affinities: int = 9,
        aff_weight: float = 1.0,
        lsds_weight: float = 1.0,
        dice_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.affinities_loss = AffinitiesLoss(
            lsds_weight=lsds_weight, affinities_weight=aff_weight, nb_affinities=nb_affinities, **kwargs
        )
        self.dice_loss = DiceLoss() if dice_weight > 0 else None
        self.dice_weight = dice_weight
        self.nb_affinities = nb_affinities

    def forward(self, output, target, mask):
        loss = self.affinities_loss(output, target, mask)

        # Optional: Add dice loss on first affinity channel (direct neighbors)
        if self.dice_loss is not None and self.dice_weight > 0:
            # Use first 3 affinity channels for dice
            dice = self.dice_loss(
                output[:, : min(3, self.nb_affinities)],
                target[:, : min(3, self.nb_affinities)],
                mask[:, : min(3, self.nb_affinities)],
            )
            loss = loss + self.dice_weight * dice

        return loss


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self, eps: float = 1e-6, foreground_factor: float = 1.0):
        super(WeightedMSELoss, self).__init__()
        self.foreground_factor = foreground_factor

    def forward(self, output, target, mask):

        scaled = mask * (output - target) ** 2

        if len(torch.nonzero(scaled)) != 0:

            masked = torch.masked_select(scaled, torch.gt(mask, 0))
            loss = torch.mean(masked)

        else:
            loss = torch.mean(scaled)

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation with class imbalance
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target, mask):
        bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        pred_prob = torch.sigmoid(output)

        # Probability of correct class
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_weight * bce_loss * mask
        return loss.sum() / mask.sum()


class BinarySegmentationLoss(nn.Module):
    """
    Combined Focal + Dice loss for binary blob segmentation
    """

    def __init__(
        self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, dice_weight: float = 0.5, focal_weight: float = 0.5
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, output, target, mask):
        focal = self.focal_loss(output, target, mask)
        dice = self.dice_loss(output, target, mask)

        return self.focal_weight * focal + self.dice_weight * dice


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

    def forward(self, input):
        x = self.unet_backbone(input)
        return self.final_conv(x)
