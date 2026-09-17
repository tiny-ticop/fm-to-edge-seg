from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fm_to_edge_seg.evaluation.robustness import (
    DEFAULT_CORRUPTIONS,
    DeterministicImageCorruption,
    evaluate_robustness,
)
from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import (
    AugmentationConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)


def test_gaussian_noise_is_reproducible_and_preserves_mask() -> None:
    image = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8))
    mask = Image.fromarray(np.zeros((16, 16), dtype=np.uint8))
    noise_spec = next(spec for spec in DEFAULT_CORRUPTIONS if spec.name == "gaussian_noise")
    transform = DeterministicImageCorruption(noise_spec, seed=42)

    first, first_mask = transform(image, mask)
    second, second_mask = transform(image, mask)

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))
    assert np.array_equal(np.asarray(first_mask), np.asarray(mask))
    assert np.array_equal(np.asarray(second_mask), np.asarray(mask))


def test_robustness_evaluation_writes_machine_readable_report() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        config = _make_config(root, manifest)
        checkpoint = root / "checkpoint.pt"
        model = MobileNetV3LiteUNet(pretrained=False)
        torch.save({"model_state": model.state_dict()}, checkpoint)

        result = evaluate_robustness(
            config,
            checkpoint,
            root / "robustness",
            split="test",
            device_name="cpu",
            condition_names=["clean", "gaussian_noise"],
            max_samples=1,
        )

        assert result.samples_per_condition == 1
        assert [item.condition for item in result.conditions] == ["clean", "gaussian_noise"]
        report = json.loads((root / "robustness" / "robustness.json").read_text())
        assert report["worst_condition"] in {"clean", "gaussian_noise"}
        assert (root / "robustness" / "robustness.csv").is_file()
        assert (root / "robustness" / "previews" / "clean.png").is_file()
        assert (root / "robustness" / "previews" / "gaussian_noise.png").is_file()


def _write_dataset(root: Path) -> Path:
    image_directory = root / "images"
    mask_directory = root / "masks"
    image_directory.mkdir()
    mask_directory.mkdir()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 180
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 1
    Image.fromarray(image).save(image_directory / "sample.png")
    Image.fromarray(mask).save(mask_directory / "sample.png")
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
                "split": "test",
            }
        )
    return manifest


def _make_config(root: Path, manifest: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="robustness_test",
        seed=42,
        output_directory=root / "training",
        data=DataConfig(manifest=manifest, input_size=(64, 64)),
        model=ModelConfig(pretrained=False),
        loss=LossConfig(bce_weight=1.0, dice_weight=1.0),
        augmentation=AugmentationConfig(False, 0.0, 0.0, 0.0, 0.0, 0.0),
        training=TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_workers=0,
            early_stopping_patience=1,
            freeze_batch_norm=True,
            device="cpu",
            mixed_precision=False,
        ),
        source_path=root / "experiment.yaml",
    )
