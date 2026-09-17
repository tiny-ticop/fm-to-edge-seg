from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.deepcrack import prepare_deepcrack
from fm_to_edge_seg.data.manifest import validate_manifest


class DeepCrackPreparationTest(unittest.TestCase):
    def test_prepares_canonical_masks_and_reproducible_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            self._make_source(source, train_count=5, test_count=2)

            first = prepare_deepcrack(source, root / "prepared_a", val_fraction=0.4, seed=7)
            second = prepare_deepcrack(source, root / "prepared_b", val_fraction=0.4, seed=7)

            self.assertEqual(first.split_counts, {"train": 3, "val": 2, "test": 2})
            self.assertTrue(validate_manifest(first.manifest_path).is_valid)
            self.assertEqual(
                self._split_map(first.manifest_path),
                self._split_map(second.manifest_path),
            )

            generated_masks = sorted((first.manifest_path.parent / "masks").rglob("*.png"))
            generated_mask = np.asarray(Image.open(generated_masks[0]))
            self.assertEqual(set(np.unique(generated_mask)), {0, 1})

    def test_rejects_unpaired_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            self._make_source(source, train_count=2, test_count=1)
            (source / "train_lab" / "train_0.png").unlink()

            with self.assertRaisesRegex(ValueError, "pairing failed"):
                prepare_deepcrack(source, root / "prepared")

    @staticmethod
    def _make_source(source: Path, train_count: int, test_count: int) -> None:
        for split, count in (("train", train_count), ("test", test_count)):
            image_directory = source / f"{split}_img"
            mask_directory = source / f"{split}_lab"
            image_directory.mkdir(parents=True)
            mask_directory.mkdir(parents=True)
            for index in range(count):
                image = np.zeros((12, 16, 3), dtype=np.uint8)
                image[:, :, index % 3] = 80
                mask = np.zeros((12, 16), dtype=np.uint8)
                mask[4:7, 2:14] = 255
                Image.fromarray(image).save(image_directory / f"{split}_{index}.jpg")
                Image.fromarray(mask).save(mask_directory / f"{split}_{index}.png")

    @staticmethod
    def _split_map(manifest_path: Path) -> dict[str, str]:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            return {row["sample_id"]: row["split"] for row in csv.DictReader(handle)}


if __name__ == "__main__":
    unittest.main()
