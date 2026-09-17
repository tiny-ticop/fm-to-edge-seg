from __future__ import annotations

import torch

from fm_to_edge_seg.losses import BinarySegmentationLoss
from fm_to_edge_seg.models import MobileNetV3LiteUNet


def test_student_preserves_input_resolution_and_exposes_features() -> None:
    model = MobileNetV3LiteUNet(pretrained=False).eval()
    image = torch.randn(1, 3, 64, 96)

    with torch.no_grad():
        logits, features = model.forward_with_features(image)

    assert logits.shape == (1, 1, 64, 96)
    assert features.encoder_s4.shape[-2:] == (16, 24)
    assert features.encoder_s8.shape[-2:] == (8, 12)
    assert features.decoder_s4.shape[-2:] == (16, 24)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_000_000


def test_loss_ignores_invalid_pixels() -> None:
    loss_function = BinarySegmentationLoss()
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[[[True, True], [False, True]]]])
    logits_a = torch.zeros_like(targets)
    logits_b = logits_a.clone()
    logits_b[0, 0, 1, 0] = 100.0

    loss_a = loss_function(logits_a, targets, valid)["total"]
    loss_b = loss_function(logits_b, targets, valid)["total"]

    assert torch.allclose(loss_a, loss_b)


def test_one_optimization_step_updates_student() -> None:
    torch.manual_seed(0)
    model = MobileNetV3LiteUNet(pretrained=False).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = BinarySegmentationLoss()
    image = torch.randn(2, 3, 64, 96)
    target = torch.zeros(2, 1, 64, 96)
    target[:, :, 20:44, 30:66] = 1.0
    valid = torch.ones_like(target, dtype=torch.bool)
    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(image), target, valid)["total"]
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight.detach())
