from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.preview import create_dataset_preview

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from fm_to_edge_seg.data.dataset import BinarySegmentationDataset


class DatasetPreviewTest(unittest.TestCase):
    def test_creates_preview_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_dataset(Path(directory))
            output = Path(directory) / "preview.png"

            result = create_dataset_preview(manifest, output, limit=1)

            self.assertEqual(result, output.resolve())
            self.assertTrue(output.is_file())
            with Image.open(output) as preview:
                self.assertEqual(preview.size, (816, 232))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_returns_aligned_tensors_and_padding_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_dataset(Path(directory))
            dataset = BinarySegmentationDataset(
                manifest_path=manifest,
                split="train",
                input_size=(16, 16),
            )

            sample = dataset[0]

            self.assertEqual(tuple(sample["image"].shape), (3, 16, 16))
            self.assertEqual(tuple(sample["mask"].shape), (1, 16, 16))
            self.assertEqual(tuple(sample["valid_mask"].shape), (1, 16, 16))
            self.assertEqual(sample["letterbox"].padding, (0, 4, 0, 4))
            self.assertFalse(sample["valid_mask"][0, 0, 0].item())
            self.assertTrue(sample["valid_mask"][0, 5, 5].item())
            self.assertGreater(sample["mask"].sum().item(), 0)

    @staticmethod
    def _write_dataset(root: Path) -> Path:
        image_directory = root / "images"
        mask_directory = root / "masks"
        image_directory.mkdir()
        mask_directory.mkdir()

        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[:, :, 1] = 100
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[4:6, 3:17] = 1
        Image.fromarray(image).save(image_directory / "sample.png")
        Image.fromarray(mask).save(mask_directory / "sample.png")

        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "image_path", "mask_path", "split"],
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


if __name__ == "__main__":
    unittest.main()
