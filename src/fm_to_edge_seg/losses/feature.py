from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeatureProjectionHead(nn.Module):
    """Training-only 1x1 projection from Student to teacher channels."""

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class DenseFeatureDistillationLoss(nn.Module):
    """Masked cosine distance between aligned dense feature maps."""

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if student_features.ndim != 4 or teacher_features.ndim != 4:
            raise ValueError("Student and teacher features must be BCHW tensors")
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(
                teacher_features,
                size=student_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        valid = F.interpolate(
            valid_mask.float(),
            size=student_features.shape[-2:],
            mode="nearest",
        ).squeeze(1)
        student_normalized = F.normalize(student_features.float(), dim=1)
        teacher_normalized = F.normalize(teacher_features.float(), dim=1)
        cosine_distance = 1.0 - (student_normalized * teacher_normalized).sum(dim=1)
        valid_count = valid.sum()
        if valid_count == 0:
            return student_features.sum() * 0.0
        return (cosine_distance * valid).sum() / valid_count
