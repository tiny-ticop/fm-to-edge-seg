from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset
from fm_to_edge_seg.data.transforms import BinarySegmentationAugmentation
from fm_to_edge_seg.distillation.dinov3_adapter import create_dinov3_feature_cache
from fm_to_edge_seg.distillation.feature_cache import (
    DenseFeatureCache,
    DenseFeatureSample,
)
from fm_to_edge_seg.distillation.feature_preview import create_feature_cache_preview
from fm_to_edge_seg.losses import DenseFeatureDistillationLoss, FeatureProjectionHead


class FakeDenseTeacher:
    @property
    def metadata(self) -> dict[str, object]:
        return {"teacher_kind": "fake_dinov3", "teacher_model": "fake_vits16"}

    def predict(self, image: Image.Image, input_size: tuple[int, int]) -> DenseFeatureSample:
        width, height = input_size
        features = np.zeros((8, height // 16, width // 16), dtype=np.float32)
        features[:, :, : features.shape[-1] // 2] = 1.0
        return DenseFeatureSample(features=features)


def test_feature_cache_and_horizontal_flip_stay_aligned() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        cache_path = create_dinov3_feature_cache(
            manifest,
            root / "feature_cache",
            teacher=FakeDenseTeacher(),
            input_size=(32, 32),
        )
        cache = DenseFeatureCache(cache_path)
        dataset = BinarySegmentationDataset(
            manifest,
            split="train",
            input_size=(32, 32),
            feature_cache=cache,
            joint_transform=BinarySegmentationAugmentation(
                horizontal_flip_probability=1.0,
                vertical_flip_probability=0.0,
                rotate_90_probability=0.0,
                brightness=0.0,
                contrast=0.0,
            ),
        )

        sample = dataset[0]

        assert sample["teacher_feature"].shape == (8, 2, 2)
        assert torch.all(sample["teacher_feature"][:, :, 0] == 0)
        assert torch.all(sample["teacher_feature"][:, :, 1] == 1)
        assert cache.metadata["partial_cache"] is False
        preview = create_feature_cache_preview(
            manifest, cache_path, root / "feature_preview.png", limit=1
        )
        assert preview.is_file()


def test_feature_projection_and_cosine_loss_backpropagate() -> None:
    torch.manual_seed(1)
    projector = FeatureProjectionHead(student_channels=4, teacher_channels=8)
    student = torch.randn(2, 4, 4, 6, requires_grad=True)
    teacher = torch.randn(2, 8, 4, 6)
    valid = torch.ones(2, 1, 16, 24, dtype=torch.bool)

    loss = DenseFeatureDistillationLoss()(projector(student), teacher, valid)
    loss.backward()

    assert 0 <= loss.item() <= 2
    assert projector.projection.weight.grad is not None
    assert student.grad is not None


def _write_dataset(root: Path) -> Path:
    (root / "images").mkdir()
    (root / "masks").mkdir()
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, :16, 1] = 180
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[4:12, 3:9] = 1
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
