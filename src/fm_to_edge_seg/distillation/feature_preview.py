from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fm_to_edge_seg.data.manifest import load_manifest
from fm_to_edge_seg.distillation.feature_cache import DenseFeatureCache


def create_feature_cache_preview(
    manifest_path: Path,
    cache_root: Path,
    output_path: Path,
    split: str = "train",
    limit: int = 6,
    seed: int = 42,
    panel_size: tuple[int, int] = (408, 288),
) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    cache = DenseFeatureCache(cache_root)
    available = [record for record in records if cache.has_sample(record.sample_id)]
    if not available:
        raise ValueError(f"Feature cache contains no samples for split '{split}'")
    random.Random(seed).shuffle(available)
    selected = available[: min(limit, len(available))]

    panel_width, panel_height = panel_size
    label_height = 24
    sheet = Image.new(
        "RGB",
        (panel_width * 2, (panel_height + label_height) * len(selected)),
        color=(28, 28, 28),
    )
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(selected):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        features = cache.load(record.sample_id).features
        pca_image = Image.fromarray(_feature_pca_rgb(features)).resize(
            image.size, Image.Resampling.NEAREST
        )
        y = row * (panel_height + label_height)
        sheet.paste(_fit_panel(image, panel_size), (0, y))
        sheet.paste(_fit_panel(pca_image, panel_size), (panel_width, y))
        draw.text((6, y + panel_height + 5), record.sample_id, fill="white")
        draw.text((panel_width + 6, y + panel_height + 5), "feature PCA", fill="white")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _feature_pca_rgb(features: np.ndarray) -> np.ndarray:
    channels, height, width = features.shape
    flattened = features.reshape(channels, -1).T.astype(np.float32)
    flattened -= flattened.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(flattened, full_matrices=False)
    component_count = min(3, right_vectors.shape[0])
    projected = flattened @ right_vectors[:component_count].T
    if component_count < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - component_count)))
    for channel in range(3):
        low, high = np.percentile(projected[:, channel], (2, 98))
        if high > low:
            projected[:, channel] = (projected[:, channel] - low) / (high - low)
        else:
            projected[:, channel] = 0
    return np.clip(projected.reshape(height, width, 3) * 255, 0, 255).astype(np.uint8)


def _fit_panel(image: Image.Image, panel_size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, panel_size, method=Image.Resampling.BILINEAR)
    panel = Image.new("RGB", panel_size, color=(0, 0, 0))
    panel.paste(
        contained,
        ((panel_size[0] - contained.width) // 2, (panel_size[1] - contained.height) // 2),
    )
    return panel
