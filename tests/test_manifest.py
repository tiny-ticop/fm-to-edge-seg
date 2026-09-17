from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.manifest import load_manifest, validate_manifest


class ManifestTest(unittest.TestCase):
    def test_valid_image_mask_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_pair(root, mask_value=1)
            manifest = self._write_manifest(root)

            records = load_manifest(manifest)
            report = validate_manifest(manifest)

            self.assertEqual(len(records), 1)
            self.assertTrue(report.is_valid, report.format())
            self.assertEqual(report.split_counts["train"], 1)

    def test_rejects_noncanonical_mask_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_pair(root, mask_value=128)
            manifest = self._write_manifest(root)

            report = validate_manifest(manifest)

            self.assertFalse(report.is_valid)
            self.assertIn("unsupported values", report.format())

    @staticmethod
    def _write_pair(root: Path, mask_value: int) -> None:
        (root / "images").mkdir()
        (root / "masks").mkdir()
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        mask = np.zeros((8, 12), dtype=np.uint8)
        mask[2:6, 3:9] = mask_value
        Image.fromarray(image, mode="RGB").save(root / "images" / "sample.png")
        Image.fromarray(mask, mode="L").save(root / "masks" / "sample.png")

    @staticmethod
    def _write_manifest(root: Path) -> Path:
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "image_path", "mask_path", "split", "group_id"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "sample",
                    "image_path": "images/sample.png",
                    "mask_path": "masks/sample.png",
                    "split": "train",
                    "group_id": "run_001",
                }
            )
        return manifest


if __name__ == "__main__":
    unittest.main()
