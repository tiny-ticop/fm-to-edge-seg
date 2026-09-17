from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from fm_to_edge_seg.deployment import benchmark_onnx, export_student_onnx
from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import (
    AugmentationConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)


def test_export_and_benchmark_onnx_student() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "student.pt"
        model = MobileNetV3LiteUNet(pretrained=False)
        torch.save({"model_state": model.state_dict()}, checkpoint)
        config = _make_config(root)

        export = export_student_onnx(
            config,
            checkpoint,
            root / "student.onnx",
            max_allowed_error=1e-4,
        )

        assert Path(export.model_path).is_file()
        assert Path(export.metadata_path).is_file()
        assert export.output_shape == (1, 1, 64, 64)
        assert export.max_absolute_error <= 1e-4
        assert len(export.model_sha256) == 64
        assert export.parameters == sum(parameter.numel() for parameter in model.parameters())

        benchmark = benchmark_onnx(
            root / "student.onnx",
            root / "benchmark.json",
            warmup_runs=1,
            measured_runs=2,
            intra_op_threads=1,
        )
        report = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
        assert benchmark.input_shape == (1, 3, 64, 64)
        assert benchmark.output_shape == (1, 1, 64, 64)
        assert benchmark.mean_milliseconds > 0
        assert benchmark.frames_per_second > 0
        assert benchmark.parameters == export.parameters
        assert report["model_sha256"] == export.model_sha256


def _make_config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="onnx_test",
        seed=42,
        output_directory=root / "training",
        data=DataConfig(manifest=root / "unused.csv", input_size=(64, 64)),
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
