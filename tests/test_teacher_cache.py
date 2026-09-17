from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset
from fm_to_edge_seg.data.transforms import BinarySegmentationAugmentation
from fm_to_edge_seg.distillation import TeacherCache, create_reference_teacher_cache


def test_reference_cache_loads_and_stays_aligned_during_flip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        cache_root = create_reference_teacher_cache(manifest, root / "cache")
        cache = TeacherCache(cache_root)
        dataset = BinarySegmentationDataset(
            manifest,
            split="train",
            input_size=(16, 16),
            teacher_cache=cache,
            joint_transform=BinarySegmentationAugmentation(
                horizontal_flip_probability=1.0,
                vertical_flip_probability=0.0,
                rotate_90_probability=0.0,
                brightness=0.0,
                contrast=0.0,
            ),
        )

        sample = dataset[0]

        assert sample["teacher_logit"].shape == sample["mask"].shape
        selected = sample["valid_mask"] & (sample["teacher_confidence"] > 0.5)
        teacher_foreground = sample["teacher_logit"] > 0
        assert np.array_equal(
            teacher_foreground[selected].numpy(), sample["mask"][selected].bool().numpy()
        )


def _write_dataset(root: Path) -> Path:
    (root / "images").mkdir()
    (root / "masks").mkdir()
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    mask = np.zeros((8, 16), dtype=np.uint8)
    mask[2:6, 1:5] = 1
    Image.fromarray(image).save(root / "images" / "sample.png")
    Image.fromarray(mask).save(root / "masks" / "sample.png")
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "image_path", "mask_path", "split"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "image_path": "images/sample.png",
                "mask_path": "masks/sample.png",
                "split": "train",
            }
        )
    return manifest
