from __future__ import annotations

import torch

from fm_to_edge_seg.metrics import BinaryMetricsAccumulator


def test_binary_metrics_ignore_padding() -> None:
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, 10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[[[True, True], [True, False]]]])
    metrics = BinaryMetricsAccumulator()

    metrics.update(logits, targets, valid)
    result = metrics.compute()

    assert abs(result["precision"] - 0.5) < 1e-7
    assert abs(result["recall"] - 1.0) < 1e-7
    assert abs(result["iou"] - 0.5) < 1e-7
    assert abs(result["dice"] - 2 / 3) < 1e-7
