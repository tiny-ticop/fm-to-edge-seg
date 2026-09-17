from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.training.config import (
    AugmentationConfig,
    DataConfig,
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
        )

        result = train_experiment(config)

        assert result.epochs_completed == 1
        assert (output / "best.pt").is_file()
        assert (output / "last.pt").is_file()
        assert (output / "history.jsonl").is_file()
        assert (output / "summary.json").is_file()
        assert (output / "validation_predictions.png").is_file()


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
