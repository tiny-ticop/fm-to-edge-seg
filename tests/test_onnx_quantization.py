from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fm_to_edge_seg.deployment import export_student_onnx, quantize_onnx_static
from fm_to_edge_seg.evaluation.onnx_evaluator import evaluate_onnx
from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import (
    AugmentationConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)


def test_static_int8_quantization_and_evaluation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_dataset(root)
        config = _make_config(root, manifest)
        checkpoint = root / "student.pt"
        model = MobileNetV3LiteUNet(pretrained=False)
        torch.save({"model_state": model.state_dict()}, checkpoint)
        export = export_student_onnx(config, checkpoint, root / "student_fp32.onnx")

        quantized = quantize_onnx_static(
            config,
            root / "student_fp32.onnx",
            root / "student_int8.onnx",
            calibration_samples=2,
        )

        assert Path(quantized.int8_model_path).is_file()
        assert Path(quantized.metadata_path).is_file()
        assert quantized.calibration_samples == 2
        assert quantized.int8_size_megabytes < quantized.fp32_size_megabytes
        assert quantized.fp32_sha256 == export.model_sha256
        assert len(quantized.int8_sha256) == 64

        evaluation = evaluate_onnx(
            config,
            root / "student_int8.onnx",
            root / "int8_evaluation",
            split="test",
            max_samples=1,
        )
        report = json.loads(
            (root / "int8_evaluation" / "metrics.json").read_text(encoding="utf-8")
        )
        assert evaluation.samples == 1
        assert 0.0 <= evaluation.dice <= 1.0
        assert report["model_sha256"] == quantized.int8_sha256
        assert (root / "int8_evaluation" / "predictions.png").is_file()


def _write_dataset(root: Path) -> Path:
    image_directory = root / "images"
    mask_directory = root / "masks"
    image_directory.mkdir()
    mask_directory.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "test")):
        y, x = np.mgrid[:64, :64]
        image = np.stack(
            (
                (x * (index + 1)) % 255,
                (y * (index + 2)) % 255,
                ((x + y) * (index + 3)) % 255,
            ),
            axis=-1,
        ).astype(np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16 + index : 48, 20:44] = 1
        filename = f"sample_{index}.png"
        Image.fromarray(image).save(image_directory / filename)
        Image.fromarray(mask).save(mask_directory / filename)
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "image_path": f"images/{filename}",
                "mask_path": f"masks/{filename}",
                "split": split,
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _make_config(root: Path, manifest: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="quantization_test",
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
