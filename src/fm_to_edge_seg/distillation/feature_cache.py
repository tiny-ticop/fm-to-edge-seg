from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseFeatureSample:
    features: np.ndarray


class DenseFeatureCache:
    """Read per-sample CxHxW dense foundation-model features."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        metadata_path = self.root / "metadata.json"
        index_path = self.root / "index.csv"
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Feature cache requires metadata.json and index.csv under {self.root}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 1:
            raise ValueError(
                f"Unsupported feature cache format: {self.metadata.get('format_version')}"
            )
        if self.metadata.get("cache_type") != "dense_features":
            raise ValueError("Cache is not a dense feature cache")
        self.feature_channels = int(self.metadata["feature_channels"])
        self.input_size = tuple(int(value) for value in self.metadata["input_size"])
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) < {"sample_id", "cache_path"}:
                raise ValueError("Feature cache index must contain sample_id and cache_path")
            self._paths = {
                row["sample_id"]: (self.root / row["cache_path"]).resolve() for row in reader
            }

    def load(self, sample_id: str) -> DenseFeatureSample:
        try:
            path = self._paths[sample_id]
        except KeyError as error:
            raise KeyError(f"Feature cache has no sample_id '{sample_id}'") from error
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Invalid feature cache path for '{sample_id}': {path}")
        with np.load(path, allow_pickle=False) as stored:
            features = stored["features"].astype(np.float32)
        _validate_feature(sample_id, features)
        if features.shape[0] != self.feature_channels:
            raise ValueError(
                f"Feature channel mismatch for '{sample_id}': "
                f"{features.shape[0]} != {self.feature_channels}"
            )
        return DenseFeatureSample(features=features)

    def validate_sample_ids(self, sample_ids: list[str]) -> None:
        missing = sorted(set(sample_ids) - self._paths.keys())
        if missing:
            raise ValueError(f"Feature cache is missing {len(missing)} samples: {missing[:10]}")

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self._paths


def write_dense_feature_cache(
    samples: Iterable[tuple[str, DenseFeatureSample]],
    output_root: Path,
    metadata: dict[str, object],
    input_size: tuple[int, int],
    overwrite: bool = False,
) -> Path:
    output_root = output_root.resolve()
    metadata_path = output_root / "metadata.json"
    index_path = output_root / "index.csv"
    if (metadata_path.exists() or index_path.exists()) and not overwrite:
        raise FileExistsError(f"Feature cache already exists at {output_root}; use --overwrite")
    samples_directory = output_root / "samples"
    samples_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    feature_channels = None
    seen_ids: set[str] = set()
    for sample_id, sample in samples:
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Empty or duplicate feature sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        _validate_feature(sample_id, sample.features)
        channels = int(sample.features.shape[0])
        if feature_channels is None:
            feature_channels = channels
        elif channels != feature_channels:
            raise ValueError(f"Inconsistent feature channels: {channels} != {feature_channels}")
        filename = hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npz"
        destination = samples_directory / filename
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace feature sample: {destination}")
        np.savez_compressed(destination, features=sample.features.astype(np.float16))
        rows.append({"sample_id": sample_id, "cache_path": f"samples/{filename}"})
    if not rows or feature_channels is None:
        raise ValueError("Cannot create an empty feature cache")

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cache_path"])
        writer.writeheader()
        writer.writerows(rows)
    complete_metadata = {
        "format_version": 1,
        "cache_type": "dense_features",
        **metadata,
        "input_size": list(input_size),
        "feature_channels": feature_channels,
        "sample_count": len(rows),
    }
    metadata_path.write_text(json.dumps(complete_metadata, indent=2), encoding="utf-8")
    return output_root


def _validate_feature(sample_id: str, features: np.ndarray) -> None:
    if features.ndim != 3 or min(features.shape) <= 0:
        raise ValueError(f"Features for '{sample_id}' must be non-empty CxHxW: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Features contain non-finite values for '{sample_id}'")
