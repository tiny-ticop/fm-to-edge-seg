from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_prediction_preview(
    normalized_images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    output_path: Path,
    sample_ids: list[str] | None = None,
) -> None:
    mean = torch.tensor((0.485, 0.456, 0.406), device=normalized_images.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=normalized_images.device).view(1, 3, 1, 1)
    images = ((normalized_images * std + mean).clamp(0, 1) * 255).byte().cpu().numpy()
    target_arrays = (targets >= 0.5).squeeze(1).cpu().numpy()
    predictions = (torch.sigmoid(logits) >= 0.5).squeeze(1).cpu().numpy()
    height, width = images.shape[-2:]
    label_height = 20
    sheet = Image.new("RGB", (width * 3, (height + label_height) * len(images)))
    draw = ImageDraw.Draw(sheet)

    for row, (image_array, target, prediction) in enumerate(
        zip(images, target_arrays, predictions, strict=True)
    ):
        image_array = image_array.transpose(1, 2, 0)
        ground_truth_overlay = _color_overlay(image_array, target, color=(255, 32, 32))
        prediction_overlay = _color_overlay(image_array, prediction, color=(32, 220, 80))
        y = row * (height + label_height)
        sheet.paste(Image.fromarray(image_array), (0, y))
        sheet.paste(Image.fromarray(ground_truth_overlay), (width, y))
        sheet.paste(Image.fromarray(prediction_overlay), (width * 2, y))
        sample_id = sample_ids[row] if sample_ids else f"sample_{row}"
        draw.text((4, y + height + 3), sample_id, fill="white")
        draw.text((width + 4, y + height + 3), "ground truth", fill="white")
        draw.text((width * 2 + 4, y + height + 3), "prediction", fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _color_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    overlay[mask] = overlay[mask] * 0.4 + np.asarray(color, dtype=np.float32) * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)
