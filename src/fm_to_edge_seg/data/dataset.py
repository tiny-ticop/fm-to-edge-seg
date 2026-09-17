from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fm_to_edge_seg.data.manifest import ManifestRecord, load_manifest

JointTransform = Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]


@dataclass(frozen=True)
class LetterboxInfo:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    scale: float


class BinarySegmentationDataset(Dataset[dict[str, Any]]):
    """Manifest-backed binary segmentation dataset with deterministic letterboxing."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        input_size: tuple[int, int],
        data_root: Path | None = None,
        joint_transform: JointTransform | None = None,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.records = [
            record
            for record in load_manifest(manifest_path, data_root=data_root)
            if record.split == split
        ]
        if not self.records:
            raise ValueError(f"Manifest contains no samples for split '{split}'")
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError("input_size must contain positive width and height")
        self.input_size = input_size
        self.joint_transform = joint_transform
        self.mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(record.mask_path) as source_mask:
            mask = source_mask.convert("L")

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        image, mask, letterbox = letterbox_pair(image, mask, self.input_size)
        image_tensor = _image_to_tensor(image)
        mask_array = np.asarray(mask, dtype=np.uint8)
        valid_array = mask_array != 255
        foreground_array = (mask_array == 1) & valid_array

        return {
            "image": (image_tensor - self.mean) / self.std,
            "mask": torch.from_numpy(foreground_array.copy()).unsqueeze(0).float(),
            "valid_mask": torch.from_numpy(valid_array.copy()).unsqueeze(0),
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "metadata": record.metadata,
            "letterbox": letterbox,
        }


def letterbox_pair(
    image: Image.Image,
    mask: Image.Image,
    target_size: tuple[int, int],
    image_fill: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, Image.Image, LetterboxInfo]:
    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch: image={image.size}, mask={mask.size}")

    target_width, target_height = target_size
    original_width, original_height = image.size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    resized_mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    right = target_width - resized_width - left
    bottom = target_height - resized_height - top

    output_image = Image.new("RGB", target_size, color=image_fill)
    output_mask = Image.new("L", target_size, color=255)
    output_image.paste(resized_image, (left, top))
    output_mask.paste(resized_mask, (left, top))
    info = LetterboxInfo(
        original_size=image.size,
        resized_size=(resized_width, resized_height),
        padding=(left, top, right, bottom),
        scale=scale,
    )
    return output_image, output_mask, info


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def records_for_split(
    manifest_path: Path,
    split: str,
    data_root: Path | None = None,
) -> list[ManifestRecord]:
    return [
        record
        for record in load_manifest(manifest_path, data_root=data_root)
        if record.split == split
    ]


def segmentation_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while preserving per-sample metadata as Python objects."""
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "sample_id": [sample["sample_id"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "mask_path": [sample["mask_path"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "letterbox": [sample["letterbox"] for sample in samples],
    }
