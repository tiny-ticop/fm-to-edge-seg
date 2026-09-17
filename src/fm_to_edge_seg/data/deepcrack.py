from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.manifest import validate_manifest

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DeepCrackPair:
    sample_id: str
    image_path: Path
    mask_path: Path
    source_split: str


@dataclass(frozen=True)
class PreparationResult:
    manifest_path: Path
    split_counts: Counter[str]


def prepare_deepcrack(
    source_root: Path,
    output_root: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> PreparationResult:
    """Convert DeepCrack into this project's canonical binary-mask layout."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    _validate_arguments(source_root, output_root, val_fraction)

    train_pairs = _discover_pairs(source_root, "train")
    test_pairs = _discover_pairs(source_root, "test")
    split_by_id = _build_splits(train_pairs, test_pairs, val_fraction, seed)

    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already contains {manifest_path}. Pass --overwrite to regenerate it."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for pair in [*train_pairs, *test_pairs]:
        split = split_by_id[pair.sample_id]
        image_destination = output_root / "images" / split / pair.image_path.name
        mask_destination = output_root / "masks" / split / f"{pair.sample_id}.png"
        _copy_image(pair.image_path, image_destination, overwrite)
        _convert_mask(pair.mask_path, mask_destination, overwrite)
        rows.append(
            {
                "sample_id": pair.sample_id,
                "image_path": image_destination.relative_to(output_root).as_posix(),
                "mask_path": mask_destination.relative_to(output_root).as_posix(),
                "split": split,
                "source_split": pair.source_split,
                "original_image": pair.image_path.name,
            }
        )

    _write_manifest(manifest_path, rows)
    report = validate_manifest(manifest_path)
    if not report.is_valid:
        raise ValueError(f"Prepared dataset failed validation:\n{report.format()}")
    return PreparationResult(manifest_path=manifest_path, split_counts=report.split_counts)


def _validate_arguments(source_root: Path, output_root: Path, val_fraction: float) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"DeepCrack source directory does not exist: {source_root}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= value < 1")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output must not be the source directory or one of its subdirectories")


def _discover_pairs(source_root: Path, split: str) -> list[DeepCrackPair]:
    image_directory = source_root / f"{split}_img"
    mask_directory = source_root / f"{split}_lab"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError(
            f"Expected {image_directory.name}/ and {mask_directory.name}/ under {source_root}"
        )

    images = _index_by_stem(image_directory)
    masks = _index_by_stem(mask_directory)
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"images without masks: {missing_masks[:10]}")
        if missing_images:
            details.append(f"masks without images: {missing_images[:10]}")
        raise ValueError("DeepCrack image/mask pairing failed; " + "; ".join(details))

    return [
        DeepCrackPair(
            sample_id=f"{split}_{stem}",
            image_path=images[stem],
            mask_path=masks[stem],
            source_split=split,
        )
        for stem in sorted(images)
    ]


def _index_by_stem(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError(f"Duplicate filename stem in {directory}: {path.stem}")
        indexed[key] = path
    if not indexed:
        raise ValueError(f"No supported image files found in {directory}")
    return indexed


def _build_splits(
    train_pairs: list[DeepCrackPair],
    test_pairs: list[DeepCrackPair],
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    shuffled_ids = [pair.sample_id for pair in train_pairs]
    random.Random(seed).shuffle(shuffled_ids)
    val_count = round(len(shuffled_ids) * val_fraction)
    if val_fraction > 0 and shuffled_ids:
        val_count = max(1, val_count)
    validation_ids = set(shuffled_ids[:val_count])

    split_by_id = {
        pair.sample_id: "val" if pair.sample_id in validation_ids else "train"
        for pair in train_pairs
    }
    split_by_id.update({pair.sample_id: "test" for pair in test_pairs})
    return split_by_id


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
        mask_array = np.asarray(mask.convert("L"))
    canonical_mask = (mask_array > 0).astype(np.uint8)
    Image.fromarray(canonical_mask).save(destination, format="PNG")


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "image_path",
        "mask_path",
        "split",
        "source_split",
        "original_image",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["sample_id"]))
