from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset
from fm_to_edge_seg.data.transforms import BinarySegmentationAugmentation
from fm_to_edge_seg.distillation import TeacherCache, create_reference_teacher_cache
from fm_to_edge_seg.distillation.cache import TeacherSample
from fm_to_edge_seg.distillation.evaluation import evaluate_teacher_cache
from fm_to_edge_seg.distillation.preview import create_teacher_cache_preview
from fm_to_edge_seg.distillation.sam3_adapter import (
    create_sam3_teacher_cache,
    merge_sam3_instances,
)


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


def test_sam3_instances_merge_into_binary_semantic_teacher() -> None:
    masks = np.zeros((2, 1, 8, 10), dtype=np.float32)
    masks[0, 0, 1:4, 2:5] = 1.0
    masks[1, 0, 3:7, 4:9] = 1.0

    sample = merge_sam3_instances(
        masks=masks,
        scores=np.array([0.8, 0.6]),
        image_size=(10, 8),
        background_confidence=0.25,
    )

    assert sample.logits.shape == (8, 10)
    assert (sample.logits > 0).sum() == 28
    assert np.isclose(sample.confidence[2, 3], 0.8)
    assert np.isclose(sample.confidence[5, 7], 0.6)
    assert np.isclose(sample.confidence[0, 0], 0.25)


def test_sam3_cache_export_is_testable_without_sam3_dependency() -> None:
    class FakeTeacher:
        @property
        def metadata(self) -> dict[str, object]:
            return {"teacher_kind": "fake_sam3", "prompt": "crack"}

        def predict(self, image: Image.Image) -> TeacherSample:
            width, height = image.size
            return TeacherSample(
                logits=np.zeros((height, width), dtype=np.float32),
                confidence=np.ones((height, width), dtype=np.float32),
            )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        output = create_sam3_teacher_cache(
            manifest,
            root / "sam3_cache",
            teacher=FakeTeacher(),
            max_samples=1,
        )

        cache = TeacherCache(output)
        assert cache.metadata["teacher_kind"] == "fake_sam3"
        assert cache.metadata["partial_cache"] is True
        assert cache.load("sample").logits.shape == (8, 16)
        preview = create_teacher_cache_preview(
            manifest, output, root / "teacher_preview.png", limit=1
        )
        assert preview.is_file()
        evaluation = evaluate_teacher_cache(
            manifest, output, root / "teacher_evaluation", confidence_threshold=0.5
        )
        assert evaluation.evaluated_samples == 1
        assert (root / "teacher_evaluation" / "summary.json").is_file()
        assert (root / "teacher_evaluation" / "per_sample.csv").is_file()


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
