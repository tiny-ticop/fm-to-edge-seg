from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fm_to_edge_seg.data.manifest import load_manifest


def create_dataset_preview(
    manifest_path: Path,
    output_path: Path,
    split: str = "train",
    limit: int = 8,
    seed: int = 42,
    panel_size: tuple[int, int] = (272, 208),
) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"Manifest contains no samples for split '{split}'")

    selected = list(records)
    random.Random(seed).shuffle(selected)
    selected = selected[: min(limit, len(selected))]
    panel_width, panel_height = panel_size
    label_height = 24
    sheet = Image.new(
        "RGB",
        (panel_width * 3, (panel_height + label_height) * len(selected)),
        color=(28, 28, 28),
    )
    draw = ImageDraw.Draw(sheet)

    for row, record in enumerate(selected):
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(record.mask_path) as source_mask:
            mask = source_mask.convert("L")
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_view = Image.fromarray(np.where(mask_array == 1, 255, 0).astype(np.uint8)).convert(
            "RGB"
        )
        overlay = _overlay_mask(image, mask_array)
        panels = [
            _fit_panel(image, panel_size),
            _fit_panel(mask_view, panel_size),
            _fit_panel(overlay, panel_size),
        ]
        y = row * (panel_height + label_height)
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * panel_width, y))
        draw.text((6, y + panel_height + 5), f"{record.sample_id} [{split}]", fill="white")
        draw.text((panel_width + 6, y + panel_height + 5), "ground truth", fill="white")
        draw.text((panel_width * 2 + 6, y + panel_height + 5), "overlay", fill="white")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _overlay_mask(image: Image.Image, mask_array: np.ndarray) -> Image.Image:
    image_array = np.asarray(image, dtype=np.float32).copy()
    foreground = mask_array == 1
    ignored = mask_array == 255
    image_array[foreground] = image_array[foreground] * 0.45 + np.array([255, 32, 32]) * 0.55
    image_array[ignored] = image_array[ignored] * 0.45 + np.array([255, 210, 0]) * 0.55
    return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))


def _fit_panel(image: Image.Image, panel_size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, panel_size, method=Image.Resampling.BILINEAR)
    panel = Image.new("RGB", panel_size, color=(0, 0, 0))
    x = (panel_size[0] - contained.width) // 2
    y = (panel_size[1] - contained.height) // 2
    panel.paste(contained, (x, y))
    return panel
