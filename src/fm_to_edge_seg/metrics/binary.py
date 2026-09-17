from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryMetricsAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        valid = valid_mask.bool()
        predictions = (torch.sigmoid(logits) >= threshold) & valid
        positives = (targets >= 0.5) & valid
        negatives = (~positives) & valid
        self.true_positive += int((predictions & positives).sum().item())
        self.false_positive += int((predictions & negatives).sum().item())
        self.false_negative += int(((~predictions) & positives).sum().item())
        self.true_negative += int(((~predictions) & negatives).sum().item())

    def compute(self) -> dict[str, float]:
        tp = float(self.true_positive)
        fp = float(self.false_positive)
        fn = float(self.false_negative)
        epsilon = 1e-8
        return {
            "iou": tp / (tp + fp + fn + epsilon),
            "dice": 2.0 * tp / (2.0 * tp + fp + fn + epsilon),
            "precision": tp / (tp + fp + epsilon),
            "recall": tp / (tp + fn + epsilon),
        }
