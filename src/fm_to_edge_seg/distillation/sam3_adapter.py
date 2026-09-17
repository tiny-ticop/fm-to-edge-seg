from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image

from fm_to_edge_seg.data.manifest import ManifestRecord, load_manifest
from fm_to_edge_seg.distillation.cache import TeacherSample, write_teacher_cache


class BinaryTeacher(Protocol):
    @property
    def metadata(self) -> dict[str, object]: ...

    def predict(self, image: Image.Image) -> TeacherSample: ...


class Sam3TextTeacher:
    """Thin adapter around the official SAM 3 text-prompt image API."""

    def __init__(
        self,
        prompt: str,
        device: str = "cuda",
        score_threshold: float = 0.5,
        foreground_probability: float = 0.99,
        background_confidence: float = 0.25,
        checkpoint_path: Path | None = None,
    ) -> None:
        if not prompt.strip():
            raise ValueError("SAM 3 text prompt must not be empty")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.5 < foreground_probability < 1.0:
            raise ValueError("foreground_probability must satisfy 0.5 < value < 1")
        if not 0.0 <= background_confidence <= 1.0:
            raise ValueError("background_confidence must be in [0, 1]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM 3 requested CUDA, but torch.cuda.is_available() is False")
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise RuntimeError(
                "Official SAM 3 is not installed. Follow docs/sam3-teacher.md in a dedicated "
                "GPU environment before running this command."
            ) from error

        build_arguments: dict[str, Any] = {"device": device, "eval_mode": True}
        if checkpoint_path is not None:
            build_arguments.update(
                checkpoint_path=str(checkpoint_path.resolve()),
                load_from_HF=False,
            )
        self.processor = Sam3Processor(
            build_sam3_image_model(**build_arguments),
            device=device,
            confidence_threshold=score_threshold,
        )
        self.prompt = prompt.strip()
        self.device = device
        self.score_threshold = score_threshold
        self.foreground_probability = foreground_probability
        self.background_confidence = background_confidence
        self.checkpoint_path = checkpoint_path
        try:
            self.sam3_version = version("sam3")
        except PackageNotFoundError:
            self.sam3_version = "unknown"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "teacher_kind": "sam3_text_prompt",
            "teacher_model": "facebookresearch/sam3",
            "sam3_package_version": self.sam3_version,
            "prompt": self.prompt,
            "score_threshold": self.score_threshold,
            "foreground_probability": self.foreground_probability,
            "background_confidence": self.background_confidence,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }

    def predict(self, image: Image.Image) -> TeacherSample:
        image = image.convert("RGB")
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=state, prompt=self.prompt)
        return merge_sam3_instances(
            masks=_as_numpy(output["masks"]),
            scores=_as_numpy(output["scores"]),
            image_size=image.size,
            foreground_probability=self.foreground_probability,
            background_confidence=self.background_confidence,
        )


def create_sam3_teacher_cache(
    manifest_path: Path,
    output_root: Path,
    teacher: BinaryTeacher,
    split: str = "train",
    max_samples: int | None = None,
    overwrite: bool = False,
) -> Path:
    manifest_path = manifest_path.resolve()
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        records = records[:max_samples]
    if not records:
        raise ValueError(f"Manifest contains no samples for split '{split}'")

    return write_teacher_cache(
        _predict_records(records, teacher),
        output_root,
        metadata={
            **teacher.metadata,
            "split": split,
            "source_manifest_sha256": _sha256_file(manifest_path),
            "partial_cache": max_samples is not None,
        },
        overwrite=overwrite,
    )


def merge_sam3_instances(
    masks: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    foreground_probability: float = 0.99,
    background_confidence: float = 0.25,
) -> TeacherSample:
    """Merge text-matched SAM 3 instances into one binary semantic target."""
    width, height = image_size
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if masks.size == 0:
        masks = np.empty((0, height, width), dtype=bool)
    elif masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError(f"Unexpected SAM 3 output: masks={masks.shape}, scores={scores.shape}")

    foreground = np.zeros((height, width), dtype=bool)
    confidence = np.full((height, width), background_confidence, dtype=np.float32)
    for mask, score in zip(masks, scores, strict=True):
        binary_mask = _binary_mask_at_size(mask, image_size)
        foreground |= binary_mask
        confidence[binary_mask] = np.maximum(
            confidence[binary_mask], np.clip(float(score), 0.0, 1.0)
        )

    low_probability = 1.0 - foreground_probability
    probability = np.where(foreground, foreground_probability, low_probability)
    logits = np.log(probability / (1.0 - probability)).astype(np.float32)
    return TeacherSample(logits=logits, confidence=confidence)


def _predict_records(
    records: list[ManifestRecord], teacher: BinaryTeacher
) -> Iterator[tuple[str, TeacherSample]]:
    total = len(records)
    for index, record in enumerate(records, start=1):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        yield record.sample_id, teacher.predict(image)
        print(f"sam3_teacher: {index}/{total} sample_id={record.sample_id}")


def _binary_mask_at_size(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    if mask.dtype == bool:
        binary = mask
    elif np.nanmin(mask) < 0 or np.nanmax(mask) > 1:
        binary = mask > 0
    else:
        binary = mask > 0.5
    width, height = image_size
    if binary.shape != (height, width):
        resized = Image.fromarray(binary.astype(np.uint8) * 255).resize(
            image_size, Image.Resampling.NEAREST
        )
        binary = np.asarray(resized) > 0
    return binary


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
