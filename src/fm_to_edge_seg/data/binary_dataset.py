from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.deepcrack import IMAGE_EXTENSIONS, PreparationResult
from fm_to_edge_seg.data.manifest import ALLOWED_SPLITS, validate_manifest


@dataclass(frozen=True)
class BinaryPair:
    sample_id: str
    image_path: Path
    mask_path: Path
    metadata: dict[str, str]


def prepare_binary_dataset(
    source_root: Path,
    output_root: Path,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> PreparationResult:
    """Create a canonical dataset from source_root/images and source_root/masks."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    _validate_arguments(source_root, output_root, val_fraction, test_fraction)
    metadata = _load_metadata(metadata_path.resolve()) if metadata_path else {}
    pairs = _discover_pairs(source_root, metadata)
    split_by_id = _build_splits(pairs, val_fraction, test_fraction, seed)

    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already contains {manifest_path}. Pass --overwrite to regenerate it."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    metadata_columns = sorted({key for pair in pairs for key in pair.metadata} - {"split"})
    for pair in pairs:
        split = split_by_id[pair.sample_id]
        image_destination = (
            output_root / "images" / split / (pair.sample_id + pair.image_path.suffix.lower())
        )
        mask_destination = output_root / "masks" / split / f"{pair.sample_id}.png"
        _copy_image(pair.image_path, image_destination, overwrite)
        _convert_mask(pair.mask_path, mask_destination, overwrite)
        row = {
            "sample_id": pair.sample_id,
            "image_path": image_destination.relative_to(output_root).as_posix(),
            "mask_path": mask_destination.relative_to(output_root).as_posix(),
            "split": split,
        }
        row.update({column: pair.metadata.get(column, "") for column in metadata_columns})
        rows.append(row)

    _write_manifest(manifest_path, rows, metadata_columns)
    report = validate_manifest(manifest_path)
    if not report.is_valid:
        raise ValueError(f"Prepared dataset failed validation:\n{report.format()}")
    return PreparationResult(manifest_path=manifest_path, split_counts=report.split_counts)


def _validate_arguments(
    source_root: Path,
    output_root: Path,
    val_fraction: float,
    test_fraction: float,
) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must satisfy 0 <= value < 1")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output must not be the source directory or one of its subdirectories")


def _discover_pairs(source_root: Path, metadata: dict[str, dict[str, str]]) -> list[BinaryPair]:
    image_directory = source_root / "images"
    mask_directory = source_root / "masks"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ under {source_root}")

    images = _index_by_relative_stem(image_directory)
    masks = _index_by_relative_stem(mask_directory)
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"images without masks: {missing_masks[:10]}")
        if missing_images:
            details.append(f"masks without images: {missing_images[:10]}")
        raise ValueError("Image/mask pairing failed; " + "; ".join(details))

    pairs = []
    for relative_stem in sorted(images):
        sample_id = relative_stem.replace("/", "__")
        pairs.append(
            BinaryPair(
                sample_id=sample_id,
                image_path=images[relative_stem],
                mask_path=masks[relative_stem],
                metadata=metadata.get(sample_id, {}),
            )
        )
    unknown_ids = sorted(metadata.keys() - {pair.sample_id for pair in pairs})
    if unknown_ids:
        raise ValueError(f"metadata.csv contains unknown sample_id values: {unknown_ids[:10]}")
    return pairs


def _index_by_relative_stem(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_stem = path.relative_to(directory).with_suffix("").as_posix()
        key = relative_stem.casefold()
        if key in indexed:
            raise ValueError(f"Duplicate relative filename stem in {directory}: {relative_stem}")
        indexed[key] = path
    if not indexed:
        raise ValueError(f"No supported image files found in {directory}")
    return indexed


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError("metadata.csv must contain a sample_id column")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            sample_id = row.pop("sample_id").strip()
            if not sample_id or sample_id in result:
                raise ValueError(f"Empty or duplicate sample_id in metadata.csv: {sample_id!r}")
            result[sample_id.casefold()] = {key: value.strip() for key, value in row.items()}
    return result


def _build_splits(
    pairs: list[BinaryPair], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    explicit = [pair.metadata.get("split", "") for pair in pairs]
    if any(explicit):
        if not all(explicit):
            raise ValueError("If metadata.csv uses split, every sample must have a split value")
        invalid = sorted(set(explicit) - ALLOWED_SPLITS)
        if invalid:
            raise ValueError(f"Unsupported explicit splits: {invalid}")
        result = {pair.sample_id: pair.metadata["split"] for pair in pairs}
        _require_train_and_val(result)
        return result

    groups: dict[str, list[str]] = {}
    for pair in pairs:
        group_id = pair.metadata.get("group_id") or pair.sample_id
        groups.setdefault(group_id, []).append(pair.sample_id)
    required_holdouts = int(val_fraction > 0) + int(test_fraction > 0)
    if len(groups) <= required_holdouts:
        raise ValueError(
            "Not enough independent groups for the requested train/val/test split. "
            "Add group_id values, provide explicit split values, or reduce holdout fractions."
        )

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    targets = {
        "test": max(1, round(len(pairs) * test_fraction)) if test_fraction > 0 else 0,
        "val": max(1, round(len(pairs) * val_fraction)) if val_fraction > 0 else 0,
    }
    assignments: dict[str, str] = {}
    remaining = group_ids.copy()
    for split in ("test", "val"):
        assigned_count = 0
        while targets[split] > assigned_count and len(remaining) > 1:
            group_id = remaining.pop(0)
            assignments[group_id] = split
            assigned_count += len(groups[group_id])
    assignments.update({group_id: "train" for group_id in remaining})
    result = {
        sample_id: assignments[group_id]
        for group_id, sample_ids in groups.items()
        for sample_id in sample_ids
    }
    _require_train_and_val(result)
    return result


def _require_train_and_val(split_by_id: dict[str, str]) -> None:
    counts = Counter(split_by_id.values())
    if not counts["train"] or not counts["val"]:
        raise ValueError("Training requires at least one train sample and one val sample")


def _copy_image(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing image: {destination}")
    shutil.copy2(source, destination)


def _convert_mask(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing mask: {destination}")
    with Image.open(source) as mask:
        array = np.asarray(mask.convert("L"))
    canonical = (array > 0).astype(np.uint8)
    Image.fromarray(canonical, mode="L").save(destination, format="PNG")


def _write_manifest(path: Path, rows: list[dict[str, str]], metadata_columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "mask_path", "split", *metadata_columns],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["sample_id"]))
