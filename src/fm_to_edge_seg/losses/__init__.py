"""Loss functions for supervised and distilled segmentation."""

from fm_to_edge_seg.losses.binary import BinarySegmentationLoss, masked_dice_loss

__all__ = ["BinarySegmentationLoss", "masked_dice_loss"]
