from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from fm_to_edge_seg.data.dataset import letterbox_pair
from fm_to_edge_seg.data.manifest import ManifestRecord, load_manifest
from fm_to_edge_seg.distillation.feature_cache import (
    DenseFeatureSample,
    write_dense_feature_cache,
)


class DenseFeatureTeacher(Protocol):
    @property
    def metadata(self) -> dict[str, object]: ...

    def predict(self, image: Image.Image, input_size: tuple[int, int]) -> DenseFeatureSample: ...


class DinoV3DenseTeacher:
    """Adapter for official DINOv3 ViT dense patch features."""

    def __init__(
        self,
        repository: Path,
        weights: Path,
        model_name: str = "dinov3_vits16",
        device: str = "cuda",
    ) -> None:
        repository = repository.resolve()
        weights = weights.resolve()
        if not repository.is_dir() or not (repository / "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv3 repository with hubconf.py not found: {repository}")
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DINOv3 requested CUDA, but torch.cuda.is_available() is False")
        self.model = torch.hub.load(
            str(repository),
            model_name,
            source="local",
            weights=str(weights),
        ).to(device)
        self.model.eval()
        self.repository = repository
        self.weights = weights
        self.model_name = model_name
        self.device = device

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "teacher_kind": "dinov3_dense_features",
            "teacher_model": self.model_name,
            "weights_filename": self.weights.name,
            "weights_sha256": _sha256_file(self.weights),
            "repository_commit": _git_commit(self.repository),
            "feature_layer": "last_normalized_patch_tokens",
            "patch_size": 16,
            "normalization": "imagenet",
        }

    def predict(self, image: Image.Image, input_size: tuple[int, int]) -> DenseFeatureSample:
        tensor = _prepare_image(image, input_size).unsqueeze(0).to(self.device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            outputs = self.model.get_intermediate_layers(tensor, n=1, reshape=True)
        feature = outputs[0]
        if isinstance(feature, (tuple, list)):
            feature = feature[-1]
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise ValueError(f"Unexpected DINOv3 feature output: {type(feature)}")
        return DenseFeatureSample(features=feature[0].float().cpu().numpy())


def create_dinov3_feature_cache(
    manifest_path: Path,
    output_root: Path,
    teacher: DenseFeatureTeacher,
    input_size: tuple[int, int],
    split: str = "train",
    max_samples: int | None = None,
    overwrite: bool = False,
) -> Path:
    if any(size <= 0 or size % 16 for size in input_size):
        raise ValueError("DINOv3 input width and height must be positive multiples of 16")
    manifest_path = manifest_path.resolve()
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        records = records[:max_samples]
    if not records:
        raise ValueError(f"Manifest contains no samples for split '{split}'")
    return write_dense_feature_cache(
        _predict_records(records, teacher, input_size),
        output_root,
        metadata={
            **teacher.metadata,
            "split": split,
            "source_manifest_sha256": _sha256_file(manifest_path),
            "partial_cache": max_samples is not None,
        },
        input_size=input_size,
        overwrite=overwrite,
    )


def _predict_records(
    records: list[ManifestRecord],
    teacher: DenseFeatureTeacher,
    input_size: tuple[int, int],
) -> Iterator[tuple[str, DenseFeatureSample]]:
    total = len(records)
    for index, record in enumerate(records, start=1):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        yield record.sample_id, teacher.predict(image, input_size)
        print(f"dinov3_teacher: {index}/{total} sample_id={record.sample_id}")


def _prepare_image(image: Image.Image, input_size: tuple[int, int]) -> torch.Tensor:
    dummy_mask = Image.new("L", image.size, color=0)
    image, _, _ = letterbox_pair(image.convert("RGB"), dummy_mask, input_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repository: Path) -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
