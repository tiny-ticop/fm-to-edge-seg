from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.binary_dataset import prepare_binary_dataset
from fm_to_edge_seg.data.manifest import load_manifest, validate_manifest


def test_prepare_binary_dataset_pairs_converts_and_groups() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "prepared"
        (source / "images").mkdir(parents=True)
        (source / "masks").mkdir()
        metadata_rows = []
        for index in range(4):
            sample_id = f"sample_{index}"
            image = np.full((12, 16, 3), index * 20, dtype=np.uint8)
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[3:9, 5:11] = 255
            Image.fromarray(image).save(source / "images" / f"{sample_id}.jpg")
            Image.fromarray(mask).save(source / "masks" / f"{sample_id}.png")
            metadata_rows.append({"sample_id": sample_id, "group_id": f"run_{index // 2}"})
        metadata = source / "metadata.csv"
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "group_id"])
            writer.writeheader()
            writer.writerows(metadata_rows)

        result = prepare_binary_dataset(
            source,
            output,
            val_fraction=0.25,
            metadata_path=metadata,
        )

        records = load_manifest(result.manifest_path)
        report = validate_manifest(result.manifest_path)
        assert report.is_valid, report.format()
        assert report.split_counts == {"train": 2, "val": 2}
        split_by_group: dict[str, set[str]] = {}
        for record in records:
            split_by_group.setdefault(record.metadata["group_id"], set()).add(record.split)
            with Image.open(record.mask_path) as mask:
                assert set(np.unique(np.asarray(mask))) == {0, 1}
        assert all(len(splits) == 1 for splits in split_by_group.values())


def test_prepare_binary_dataset_requires_enough_groups() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        (source / "images").mkdir(parents=True)
        (source / "masks").mkdir()
        Image.new("RGB", (8, 8)).save(source / "images" / "only.png")
        Image.new("L", (8, 8), color=255).save(source / "masks" / "only.png")

        try:
            prepare_binary_dataset(source, root / "prepared", val_fraction=0.2)
        except ValueError as error:
            assert "Not enough independent groups" in str(error)
        else:
            raise AssertionError("Expected a small-dataset split error")
