from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    input_size: tuple[int, int]


@dataclass(frozen=True)
class ModelConfig:
    pretrained: bool


@dataclass(frozen=True)
class LossConfig:
    bce_weight: float
    dice_weight: float


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    horizontal_flip_probability: float
    vertical_flip_probability: float
    rotate_90_probability: float
    brightness: float
    contrast: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    early_stopping_patience: int
    freeze_batch_norm: bool
    device: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int
    output_directory: Path
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    source_path: Path


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = path.resolve()
    root = _find_project_root(path.parent)
    experiment = _load_yaml(path)
    data = _load_yaml(_resolve_reference(root, experiment["data"]))
    model = _load_yaml(_resolve_reference(root, experiment["model"]))
    loss = _load_yaml(_resolve_reference(root, experiment["loss"]))
    training = experiment.get("training", {})
    augmentation = experiment.get("augmentation", {})
    input_config = data["input"]

    config = ExperimentConfig(
        experiment_id=str(experiment["experiment_id"]),
        seed=int(experiment.get("seed", 42)),
        output_directory=_resolve_reference(root, experiment["output_dir"]),
        data=DataConfig(
            manifest=_resolve_reference(root, data["manifest"]),
            input_size=(int(input_config["width"]), int(input_config["height"])),
        ),
        model=ModelConfig(pretrained=bool(model.get("encoder_pretrained", True))),
        loss=LossConfig(
            bce_weight=float(loss.get("bce_weight", 1.0)),
            dice_weight=float(loss.get("dice_weight", 1.0)),
        ),
        augmentation=AugmentationConfig(
            enabled=bool(augmentation.get("enabled", False)),
            horizontal_flip_probability=float(augmentation.get("horizontal_flip_probability", 0.5)),
            vertical_flip_probability=float(augmentation.get("vertical_flip_probability", 0.5)),
            rotate_90_probability=float(augmentation.get("rotate_90_probability", 0.5)),
            brightness=float(augmentation.get("brightness", 0.1)),
            contrast=float(augmentation.get("contrast", 0.1)),
        ),
        training=TrainingConfig(
            epochs=int(training.get("epochs", 50)),
            batch_size=int(training.get("batch_size", 8)),
            learning_rate=float(training.get("learning_rate", 3e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            num_workers=int(training.get("num_workers", 2)),
            early_stopping_patience=int(training.get("early_stopping_patience", 10)),
            freeze_batch_norm=bool(training.get("freeze_batch_norm", False)),
            device=str(training.get("device", "auto")),
        ),
        source_path=path,
    )
    _validate_config(config)
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _resolve_reference(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find pyproject.toml above {start}")


def _validate_config(config: ExperimentConfig) -> None:
    if config.training.epochs <= 0 or config.training.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not config.data.manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {config.data.manifest}")
