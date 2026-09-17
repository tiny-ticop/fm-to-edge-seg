"""Loss functions for supervised and distilled segmentation."""

from fm_to_edge_seg.losses.binary import (
    BinaryLogitDistillationLoss,
    BinarySegmentationLoss,
    masked_dice_loss,
)

__all__ = ["BinaryLogitDistillationLoss", "BinarySegmentationLoss", "masked_dice_loss"]
