"""Loss functions for supervised and distilled segmentation."""

from fm_to_edge_seg.losses.binary import (
    BinaryLogitDistillationLoss,
    BinarySegmentationLoss,
    masked_dice_loss,
)
from fm_to_edge_seg.losses.feature import DenseFeatureDistillationLoss, FeatureProjectionHead

__all__ = [
    "BinaryLogitDistillationLoss",
    "BinarySegmentationLoss",
    "DenseFeatureDistillationLoss",
    "FeatureProjectionHead",
    "masked_dice_loss",
]
