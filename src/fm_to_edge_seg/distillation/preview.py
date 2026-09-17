from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fm_to_edge_seg.data.manifest import load_manifest
from fm_to_edge_seg.distillation.cache import TeacherCache


def create_teacher_cache_preview(
    manifest_path: Path,
    cache_root: Path,
    output_path: Path,
    split: str = "train",
    limit: int = 8,
    seed: int = 42,
    panel_size: tuple[int, int] = (272, 208),
) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    cache = TeacherCache(cache_root)
    available = [record for record in records if cache.has_sample(record.sample_id)]
    if not available:
        raise ValueError(f"Teacher cache contains no samples for split '{split}'")
    random.Random(seed).shuffle(available)
    selected = available[: min(limit, len(available))]

    panel_width, panel_height = panel_size
    label_height = 24
    sheet = Image.new(
        "RGB",
        (panel_width * 4, (panel_height + label_height) * len(selected)),
        color=(28, 28, 28),
    )
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(selected):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        with Image.open(record.mask_path) as source:
            ground_truth = np.asarray(source.convert("L")) == 1
        teacher = cache.load(record.sample_id)
        if teacher.logits.shape != (image.height, image.width):
            raise ValueError(f"Teacher/image size mismatch for '{record.sample_id}'")
        prediction = teacher.logits > 0
        confidence = Image.fromarray(
            np.clip(teacher.confidence * 255, 0, 255).astype(np.uint8), mode="L"
        ).convert("RGB")
        panels = [
            image,
            _overlay(image, ground_truth, (255, 32, 32)),
            _overlay(image, prediction, (32, 220, 80)),
            confidence,
        ]
        y = row * (panel_height + label_height)
        for column, panel in enumerate(panels):
            sheet.paste(_fit_panel(panel, panel_size), (column * panel_width, y))
        labels = [record.sample_id, "ground truth", "teacher", "confidence"]
        for column, label in enumerate(labels):
            draw.text((column * panel_width + 6, y + panel_height + 5), label, fill="white")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    array = np.asarray(image, dtype=np.float32).copy()
    array[mask] = array[mask] * 0.4 + np.asarray(color) * 0.6
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _fit_panel(image: Image.Image, panel_size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, panel_size, method=Image.Resampling.BILINEAR)
    panel = Image.new("RGB", panel_size, color=(0, 0, 0))
    panel.paste(
        contained,
        ((panel_size[0] - contained.width) // 2, (panel_size[1] - contained.height) // 2),
    )
    return panel
