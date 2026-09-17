from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.manifest import load_manifest


@dataclass(frozen=True)
class TeacherSample:
    logits: np.ndarray
    confidence: np.ndarray


class TeacherCache:
    """Read model-independent, per-sample binary teacher logits."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        metadata_path = self.root / "metadata.json"
        index_path = self.root / "index.csv"
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Teacher cache requires metadata.json and index.csv under {self.root}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 1:
            raise ValueError(
                f"Unsupported teacher cache format: {self.metadata.get('format_version')}"
            )
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) < {"sample_id", "cache_path"}:
                raise ValueError("Teacher cache index must contain sample_id and cache_path")
            self._paths = {
                row["sample_id"]: (self.root / row["cache_path"]).resolve() for row in reader
            }

    def load(self, sample_id: str) -> TeacherSample:
        try:
            path = self._paths[sample_id]
        except KeyError as error:
            raise KeyError(f"Teacher cache has no sample_id '{sample_id}'") from error
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Invalid teacher cache path for '{sample_id}': {path}")
        with np.load(path, allow_pickle=False) as stored:
            logits = stored["logits"].astype(np.float32)
            confidence = stored["confidence"].astype(np.float32)
        if logits.ndim != 2 or confidence.shape != logits.shape:
            raise ValueError(
                f"Teacher arrays for '{sample_id}' must be equal HxW arrays, got "
                f"logits={logits.shape}, confidence={confidence.shape}"
            )
        if not np.isfinite(logits).all() or not np.isfinite(confidence).all():
            raise ValueError(f"Teacher cache contains non-finite values for '{sample_id}'")
        if confidence.min() < 0 or confidence.max() > 1:
            raise ValueError(f"Teacher confidence must be in [0, 1] for '{sample_id}'")
        return TeacherSample(logits=logits, confidence=confidence)

    def validate_sample_ids(self, sample_ids: list[str]) -> None:
        missing = sorted(set(sample_ids) - self._paths.keys())
        if missing:
            raise ValueError(f"Teacher cache is missing {len(missing)} samples: {missing[:10]}")

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self._paths


def create_reference_teacher_cache(
    manifest_path: Path,
    output_root: Path,
    split: str = "train",
    foreground_probability: float = 0.99,
    overwrite: bool = False,
) -> Path:
    """Build a test-only teacher cache from labels, not from a foundation model."""
    if not 0.5 < foreground_probability < 1.0:
        raise ValueError("foreground_probability must satisfy 0.5 < value < 1")
    manifest_path = manifest_path.resolve()
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"Manifest contains no samples for split '{split}'")
    probability_low = 1.0 - foreground_probability

    def samples() -> Iterable[tuple[str, TeacherSample]]:
        for record in records:
            with Image.open(record.mask_path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8)
            valid = mask != 255
            probability = np.where(mask == 1, foreground_probability, probability_low)
            probability = np.clip(probability, 1e-6, 1 - 1e-6)
            logits = np.log(probability / (1.0 - probability)).astype(np.float32)
            yield (
                record.sample_id,
                TeacherSample(
                    logits=logits,
                    confidence=valid.astype(np.float32),
                ),
            )

    return write_teacher_cache(
        samples(),
        output_root,
        metadata={
            "teacher_kind": "reference_ground_truth_mask",
            "split": split,
            "source_manifest_sha256": _sha256_file(manifest_path),
            "warning": "Pipeline verification only; this is not a foundation-model result.",
        },
        overwrite=overwrite,
    )


def write_teacher_cache(
    samples: Iterable[tuple[str, TeacherSample]],
    output_root: Path,
    metadata: dict[str, object],
    overwrite: bool = False,
) -> Path:
    """Write teacher predictions using the common cache interchange format."""
    output_root = output_root.resolve()
    metadata_path = output_root / "metadata.json"
    index_path = output_root / "index.csv"
    if (metadata_path.exists() or index_path.exists()) and not overwrite:
        raise FileExistsError(f"Teacher cache already exists at {output_root}; use --overwrite")
    samples_directory = output_root / "samples"
    samples_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_ids: set[str] = set()
    for sample_id, sample in samples:
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Empty or duplicate teacher sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        _validate_teacher_sample(sample_id, sample)
        filename = hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npz"
        destination = samples_directory / filename
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace teacher sample: {destination}")
        np.savez_compressed(
            destination,
            logits=sample.logits.astype(np.float16),
            confidence=sample.confidence.astype(np.float16),
        )
        rows.append({"sample_id": sample_id, "cache_path": f"samples/{filename}"})
    if not rows:
        raise ValueError("Cannot create an empty teacher cache")

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cache_path"])
        writer.writeheader()
        writer.writerows(rows)
    complete_metadata = {"format_version": 1, **metadata, "sample_count": len(rows)}
    metadata_path.write_text(
        json.dumps(complete_metadata, indent=2),
        encoding="utf-8",
    )
    return output_root


def _validate_teacher_sample(sample_id: str, sample: TeacherSample) -> None:
    if sample.logits.ndim != 2 or sample.confidence.shape != sample.logits.shape:
        raise ValueError(
            f"Teacher arrays for '{sample_id}' must be equal HxW arrays, got "
            f"logits={sample.logits.shape}, confidence={sample.confidence.shape}"
        )
    if not np.isfinite(sample.logits).all() or not np.isfinite(sample.confidence).all():
        raise ValueError(f"Teacher sample contains non-finite values for '{sample_id}'")
    if sample.confidence.min() < 0 or sample.confidence.max() > 1:
        raise ValueError(f"Teacher confidence must be in [0, 1] for '{sample_id}'")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
