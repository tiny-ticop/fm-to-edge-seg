from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BinarySegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_shapes(logits, targets, valid_mask)
        valid = (
            torch.ones_like(targets, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        )
        valid_float = valid.float()
        valid_count = valid_float.sum().clamp_min(1.0)

        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (bce_map * valid_float).sum() / valid_count
        dice = masked_dice_loss(logits, targets, valid)
        total = self.bce_weight * bce + self.dice_weight * dice
        return {"total": total, "bce": bce, "dice": dice}


def masked_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1.0,
) -> torch.Tensor:
    valid = torch.ones_like(targets, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    valid_float = valid.float()
    probabilities = torch.sigmoid(logits) * valid_float
    targets = targets * valid_float
    reduce_dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * targets).sum(dim=reduce_dimensions)
    denominator = probabilities.sum(dim=reduce_dimensions) + targets.sum(dim=reduce_dimensions)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def _validate_shapes(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must match: {logits.shape} != {targets.shape}")
    if valid_mask is not None and valid_mask.shape != targets.shape:
        raise ValueError(
            f"valid_mask and targets must match: {valid_mask.shape} != {targets.shape}"
        )
