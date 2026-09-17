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
from fm_to_edge_seg.distillation.cache import TeacherCache
from fm_to_edge_seg.distillation.feature_cache import DenseFeatureCache

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
        teacher_cache: TeacherCache | None = None,
        feature_cache: DenseFeatureCache | None = None,
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
        self.teacher_cache = teacher_cache
        self.feature_cache = feature_cache
        if teacher_cache is not None:
            teacher_cache.validate_sample_ids([record.sample_id for record in self.records])
        if feature_cache is not None:
            feature_cache.validate_sample_ids([record.sample_id for record in self.records])
            if feature_cache.input_size != input_size:
                raise ValueError(
                    f"Feature cache input_size {feature_cache.input_size} does not match "
                    f"dataset input_size {input_size}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(record.mask_path) as source_mask:
            mask = source_mask.convert("L")

        dense_targets: list[Image.Image] | None = None
        teacher_feature: torch.Tensor | None = None
        if self.teacher_cache is not None:
            teacher = self.teacher_cache.load(record.sample_id)
            if teacher.logits.shape != (image.height, image.width):
                raise ValueError(
                    f"Teacher/image size mismatch for '{record.sample_id}': "
                    f"teacher={teacher.logits.shape[::-1]}, image={image.size}"
                )
            teacher_probability = 1.0 / (1.0 + np.exp(-teacher.logits))
            dense_targets = [
                Image.fromarray(teacher_probability.astype(np.float32), mode="F"),
                Image.fromarray(teacher.confidence.astype(np.float32), mode="F"),
            ]
        if self.feature_cache is not None:
            feature_sample = self.feature_cache.load(record.sample_id)
            teacher_feature = torch.from_numpy(feature_sample.features.copy())

        if self.joint_transform is not None:
            if dense_targets is None and teacher_feature is None:
                image, mask = self.joint_transform(image, mask)
            elif teacher_feature is not None:
                image, mask, transformed_targets, teacher_feature = self.joint_transform(
                    image,
                    mask,
                    dense_targets,
                    teacher_feature,
                )
                dense_targets = transformed_targets or None
            else:
                image, mask, dense_targets = self.joint_transform(image, mask, dense_targets)

        original_size = image.size
        image, mask, letterbox = letterbox_pair(image, mask, self.input_size)
        image_tensor = _image_to_tensor(image)
        mask_array = np.asarray(mask, dtype=np.uint8)
        valid_array = mask_array != 255
        foreground_array = (mask_array == 1) & valid_array

        sample = {
            "image": (image_tensor - self.mean) / self.std,
            "mask": torch.from_numpy(foreground_array.copy()).unsqueeze(0).float(),
            "valid_mask": torch.from_numpy(valid_array.copy()).unsqueeze(0),
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "metadata": record.metadata,
            "letterbox": letterbox,
        }
        if dense_targets is not None:
            probability_image = _letterbox_dense_target(
                dense_targets[0], original_size, letterbox, fill=0.5
            )
            confidence_image = _letterbox_dense_target(
                dense_targets[1], original_size, letterbox, fill=0.0
            )
            probability = np.clip(np.asarray(probability_image), 1e-6, 1 - 1e-6)
            teacher_logit = np.log(probability / (1.0 - probability)).astype(np.float32)
            teacher_confidence = np.asarray(confidence_image, dtype=np.float32).copy()
            sample["teacher_logit"] = torch.from_numpy(teacher_logit).unsqueeze(0)
            sample["teacher_confidence"] = torch.from_numpy(teacher_confidence).unsqueeze(0)
        if teacher_feature is not None:
            sample["teacher_feature"] = teacher_feature
        return sample


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


def _letterbox_dense_target(
    target: Image.Image,
    expected_size: tuple[int, int],
    letterbox: LetterboxInfo,
    fill: float,
) -> Image.Image:
    if target.size != expected_size:
        raise ValueError(f"Dense target size mismatch: {target.size} != {expected_size}")
    resized = target.resize(letterbox.resized_size, Image.Resampling.BILINEAR)
    width = letterbox.resized_size[0] + letterbox.padding[0] + letterbox.padding[2]
    height = letterbox.resized_size[1] + letterbox.padding[1] + letterbox.padding[3]
    output = Image.new("F", (width, height), color=fill)
    output.paste(resized, (letterbox.padding[0], letterbox.padding[1]))
    return output


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
    batch = {
        "image": torch.stack([sample["image"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "sample_id": [sample["sample_id"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "mask_path": [sample["mask_path"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "letterbox": [sample["letterbox"] for sample in samples],
    }
    if "teacher_logit" in samples[0]:
        if not all("teacher_logit" in sample for sample in samples):
            raise ValueError("Teacher tensors must be present for every sample in a batch")
        batch["teacher_logit"] = torch.stack([sample["teacher_logit"] for sample in samples])
        batch["teacher_confidence"] = torch.stack(
            [sample["teacher_confidence"] for sample in samples]
        )
    if "teacher_feature" in samples[0]:
        if not all("teacher_feature" in sample for sample in samples):
            raise ValueError("Teacher features must be present for every sample in a batch")
        batch["teacher_feature"] = torch.stack([sample["teacher_feature"] for sample in samples])
    return batch
