from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fm_to_edge_seg.distillation import create_reference_teacher_cache
from fm_to_edge_seg.distillation.feature_cache import (
    DenseFeatureSample,
    write_dense_feature_cache,
)
from fm_to_edge_seg.evaluation.evaluator import evaluate_checkpoint
from fm_to_edge_seg.training.config import (
    AugmentationConfig,
    DataConfig,
    DistillationConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from fm_to_edge_seg.training.trainer import train_experiment


def test_training_smoke_creates_reproducibility_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        teacher_cache = create_reference_teacher_cache(manifest, root / "teacher_cache")
        output = root / "artifacts"
        config = ExperimentConfig(
            experiment_id="test_smoke",
            seed=3,
            output_directory=output,
            data=DataConfig(manifest=manifest, input_size=(64, 64)),
            model=ModelConfig(pretrained=False),
            loss=LossConfig(bce_weight=1.0, dice_weight=1.0),
            augmentation=AugmentationConfig(
                enabled=False,
                horizontal_flip_probability=0.0,
                vertical_flip_probability=0.0,
                rotate_90_probability=0.0,
                brightness=0.0,
                contrast=0.0,
            ),
            training=TrainingConfig(
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                num_workers=0,
                early_stopping_patience=2,
                freeze_batch_norm=True,
                device="cpu",
            ),
            source_path=root / "experiment.yaml",
            distillation=DistillationConfig(
                teacher="reference_ground_truth_mask",
                cache_root=teacher_cache,
                temperature=2.0,
                weight=0.5,
                confidence_threshold=0.0,
            ),
        )

        result = train_experiment(config)

        assert result.epochs_completed == 1
        assert (output / "best.pt").is_file()
        assert (output / "last.pt").is_file()
        assert (output / "history.jsonl").is_file()
        assert (output / "summary.json").is_file()
        assert (output / "validation_predictions.png").is_file()

        evaluation_output = root / "evaluation"
        evaluation = evaluate_checkpoint(
            config,
            output / "best.pt",
            evaluation_output,
            split="val",
            device_name="cpu",
        )
        assert evaluation.samples == 1
        assert (evaluation_output / "metrics.json").is_file()
        assert (evaluation_output / "predictions.png").is_file()


def test_feature_distillation_training_smoke() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        feature_cache = write_dense_feature_cache(
            (
                (
                    f"sample_{index}",
                    DenseFeatureSample(features=np.full((8, 4, 4), index + 1, dtype=np.float32)),
                )
                for index in range(2)
            ),
            root / "feature_cache",
            metadata={"teacher_kind": "fake_dinov3"},
            input_size=(64, 64),
        )
        output = root / "feature_artifacts"
        config = ExperimentConfig(
            experiment_id="test_feature_kd",
            seed=4,
            output_directory=output,
            data=DataConfig(manifest=manifest, input_size=(64, 64)),
            model=ModelConfig(pretrained=False),
            loss=LossConfig(bce_weight=1.0, dice_weight=1.0),
            augmentation=AugmentationConfig(
                enabled=False,
                horizontal_flip_probability=0.0,
                vertical_flip_probability=0.0,
                rotate_90_probability=0.0,
                brightness=0.0,
                contrast=0.0,
            ),
            training=TrainingConfig(
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                num_workers=0,
                early_stopping_patience=2,
                freeze_batch_norm=True,
                device="cpu",
            ),
            source_path=root / "experiment.yaml",
            distillation=DistillationConfig(
                teacher="fake_dinov3",
                cache_root=feature_cache,
                temperature=1.0,
                weight=0.25,
                confidence_threshold=0.0,
                kind="feature",
                student_feature="encoder_s16",
            ),
        )

        result = train_experiment(config)

        assert result.epochs_completed == 1
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        assert "feature_projector_state" in checkpoint


def _write_dataset(root: Path) -> Path:
    image_directory = root / "images"
    mask_directory = root / "masks"
    image_directory.mkdir()
    mask_directory.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, index % 3] = 120
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 24:40] = 1
        image_name = f"sample_{index}.png"
        Image.fromarray(image).save(image_directory / image_name)
        Image.fromarray(mask).save(mask_directory / image_name)
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "image_path": f"images/{image_name}",
                "mask_path": f"masks/{image_name}",
                "split": split,
            }
        )

    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return manifest
