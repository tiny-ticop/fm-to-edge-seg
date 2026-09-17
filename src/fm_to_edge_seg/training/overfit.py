from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset
from fm_to_edge_seg.losses import BinarySegmentationLoss
from fm_to_edge_seg.models import MobileNetV3LiteUNet


@dataclass(frozen=True)
class OverfitResult:
    initial_loss: float
    final_loss: float
    relative_loss: float
    initial_dice: float
    final_dice: float
    steps: int
    samples: int
    elapsed_seconds: float
    output_directory: str


def run_overfit_experiment(
    manifest_path: Path,
    output_directory: Path,
    split: str = "train",
    input_size: tuple[int, int] = (272, 192),
    sample_count: int = 2,
    steps: int = 30,
    learning_rate: float = 3e-3,
    seed: int = 42,
    pretrained: bool = False,
    device_name: str = "cpu",
) -> OverfitResult:
    if sample_count <= 0 or steps <= 0:
        raise ValueError("sample_count and steps must be positive")
    _set_seed(seed)
    device = torch.device(device_name)
    dataset = BinarySegmentationDataset(manifest_path, split=split, input_size=input_size)
    selected_count = min(sample_count, len(dataset))
    samples = [dataset[index] for index in range(selected_count)]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    masks = torch.stack([sample["mask"] for sample in samples]).to(device)
    valid_masks = torch.stack([sample["valid_mask"] for sample in samples]).to(device)

    model = MobileNetV3LiteUNet(pretrained=pretrained).to(device)
    loss_function = BinarySegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    initial_loss = 0.0
    initial_dice = 0.0
    started_at = time.perf_counter()
    model.train()
    _freeze_batch_norm_statistics(model)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        losses = loss_function(logits, masks, valid_masks)
        losses["total"].backward()
        optimizer.step()

        current_loss = float(losses["total"].detach().cpu())
        current_dice = _hard_dice(logits.detach(), masks, valid_masks)
        if step == 0:
            initial_loss = current_loss
            initial_dice = current_dice
        if step == steps - 1 or step % max(1, steps // 5) == 0:
            print(
                f"step={step + 1:04d}/{steps:04d} loss={current_loss:.6f} dice={current_dice:.4f}"
            )

    elapsed_seconds = time.perf_counter() - started_at
    model.eval()
    with torch.no_grad():
        final_logits = model(images)
        final_losses = loss_function(final_logits, masks, valid_masks)
        final_loss = float(final_losses["total"].cpu())
        final_dice = _hard_dice(final_logits, masks, valid_masks)

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    result = OverfitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        relative_loss=final_loss / max(initial_loss, 1e-12),
        initial_dice=initial_dice,
        final_dice=final_dice,
        steps=steps,
        samples=selected_count,
        elapsed_seconds=elapsed_seconds,
        output_directory=str(output_directory),
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "result": asdict(result),
        },
        output_directory / "checkpoint.pt",
    )
    _save_prediction_preview(images, masks, final_logits, output_directory / "predictions.png")
    (output_directory / "metrics.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    return result


def _hard_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    valid = valid_mask.bool()
    predictions = (torch.sigmoid(logits) >= threshold) & valid
    targets_binary = (targets >= 0.5) & valid
    intersection = (predictions & targets_binary).sum().float()
    denominator = predictions.sum().float() + targets_binary.sum().float()
    return float(((2.0 * intersection + 1.0) / (denominator + 1.0)).cpu())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _freeze_batch_norm_statistics(model: nn.Module) -> None:
    """Use stable running statistics while retaining gradients for affine parameters."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def _save_prediction_preview(
    normalized_images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    output_path: Path,
) -> None:
    mean = torch.tensor((0.485, 0.456, 0.406), device=normalized_images.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=normalized_images.device).view(1, 3, 1, 1)
    images = ((normalized_images * std + mean).clamp(0, 1) * 255).byte().cpu().numpy()
    target_arrays = (targets >= 0.5).squeeze(1).cpu().numpy()
    predictions = (torch.sigmoid(logits) >= 0.5).squeeze(1).cpu().numpy()
    height, width = images.shape[-2:]
    sheet = Image.new("RGB", (width * 3, height * len(images)))

    for row, (image_array, target, prediction) in enumerate(
        zip(images, target_arrays, predictions, strict=True)
    ):
        image_array = image_array.transpose(1, 2, 0)
        ground_truth_overlay = _color_overlay(image_array, target, color=(255, 32, 32))
        prediction_overlay = _color_overlay(image_array, prediction, color=(32, 220, 80))
        y = row * height
        sheet.paste(Image.fromarray(image_array), (0, y))
        sheet.paste(Image.fromarray(ground_truth_overlay), (width, y))
        sheet.paste(Image.fromarray(prediction_overlay), (width * 2, y))
    sheet.save(output_path)


def _color_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    overlay[mask] = overlay[mask] * 0.4 + np.asarray(color, dtype=np.float32) * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)
