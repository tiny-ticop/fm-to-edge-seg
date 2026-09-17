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


class BinaryLogitDistillationLoss(nn.Module):
    """Bernoulli KL divergence between teacher and student predictions."""

    def __init__(self, temperature: float = 2.0, confidence_threshold: float = 0.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        _validate_shapes(student_logits, teacher_logits, valid_mask)
        if confidence.shape != student_logits.shape:
            raise ValueError(
                f"confidence and logits must match: {confidence.shape} != {student_logits.shape}"
            )
        selected = valid_mask.bool() & (confidence >= self.confidence_threshold)
        selected_float = selected.float()
        selected_count = selected_float.sum()
        if selected_count == 0:
            return student_logits.sum() * 0.0

        student_scaled = student_logits / self.temperature
        teacher_scaled = teacher_logits / self.temperature
        teacher_probability = torch.sigmoid(teacher_scaled)
        kl_map = teacher_probability * (
            F.logsigmoid(teacher_scaled) - F.logsigmoid(student_scaled)
        ) + (1.0 - teacher_probability) * (
            F.logsigmoid(-teacher_scaled) - F.logsigmoid(-student_scaled)
        )
        return (kl_map * selected_float).sum() / selected_count * self.temperature**2


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
