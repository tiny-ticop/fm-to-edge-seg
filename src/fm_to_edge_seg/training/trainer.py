from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset, segmentation_collate
from fm_to_edge_seg.data.transforms import BinarySegmentationAugmentation
from fm_to_edge_seg.evaluation import save_prediction_preview
from fm_to_edge_seg.losses import BinarySegmentationLoss
from fm_to_edge_seg.metrics import BinaryMetricsAccumulator
from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import ExperimentConfig


@dataclass(frozen=True)
class EpochResult:
    loss: float
    bce: float
    dice_loss: float
    iou: float
    dice: float
    precision: float
    recall: float
    samples: int


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_dice: float
    epochs_completed: int
    elapsed_seconds: float
    output_directory: str


def train_experiment(
    config: ExperimentConfig,
    device_override: str | None = None,
    epochs_override: int | None = None,
    num_workers_override: int | None = None,
    pretrained_override: bool | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> TrainingResult:
    training = config.training
    if epochs_override is not None:
        training = replace(training, epochs=epochs_override)
    if num_workers_override is not None:
        training = replace(training, num_workers=num_workers_override)
    model_config = config.model
    if pretrained_override is not None:
        model_config = replace(model_config, pretrained=pretrained_override)

    _set_seed(config.seed)
    device = _select_device(device_override or training.device)
    output_directory = config.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(config, output_directory, device, training, model_config.pretrained)

    train_transform = None
    if config.augmentation.enabled:
        train_transform = BinarySegmentationAugmentation(
            horizontal_flip_probability=config.augmentation.horizontal_flip_probability,
            vertical_flip_probability=config.augmentation.vertical_flip_probability,
            rotate_90_probability=config.augmentation.rotate_90_probability,
            brightness=config.augmentation.brightness,
            contrast=config.augmentation.contrast,
        )
    train_dataset = BinarySegmentationDataset(
        config.data.manifest,
        split="train",
        input_size=config.data.input_size,
        joint_transform=train_transform,
    )
    validation_dataset = BinarySegmentationDataset(
        config.data.manifest,
        split="val",
        input_size=config.data.input_size,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        seed=config.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        seed=config.seed,
        pin_memory=device.type == "cuda",
    )

    model = MobileNetV3LiteUNet(pretrained=model_config.pretrained).to(device)
    loss_function = BinarySegmentationLoss(
        bce_weight=config.loss.bce_weight,
        dice_weight=config.loss.dice_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, training.epochs),
    )

    best_validation_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_path = output_directory / "history.jsonl"
    history_path.unlink(missing_ok=True)
    started_at = time.perf_counter()
    epochs_completed = 0

    for epoch in range(1, training.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            freeze_batch_norm=training.freeze_batch_norm,
            max_batches=max_train_batches,
        )
        validation_metrics = _run_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
            freeze_batch_norm=False,
            max_batches=max_validation_batches,
        )
        scheduler.step()
        epochs_completed = epoch
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")
        print(
            f"epoch={epoch:03d}/{training.epochs:03d} "
            f"train_loss={train_metrics.loss:.4f} train_dice={train_metrics.dice:.4f} "
            f"val_loss={validation_metrics.loss:.4f} val_dice={validation_metrics.dice:.4f} "
            f"val_iou={validation_metrics.iou:.4f}"
        )

        _save_checkpoint(
            output_directory / "last.pt",
            model,
            optimizer,
            epoch,
            validation_metrics,
            config,
        )
        if validation_metrics.dice > best_validation_dice:
            best_validation_dice = validation_metrics.dice
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                output_directory / "best.pt",
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training.early_stopping_patience:
            print(
                "early_stopping: no validation Dice improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    elapsed_seconds = time.perf_counter() - started_at
    best_checkpoint = torch.load(
        output_directory / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["model_state"])
    _save_validation_preview(model, validation_loader, device, output_directory)

    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_dice=best_validation_dice,
        epochs_completed=epochs_completed,
        elapsed_seconds=elapsed_seconds,
        output_directory=str(output_directory),
    )
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    return result


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_function: BinarySegmentationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    freeze_batch_norm: bool,
    max_batches: int | None,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)
    if is_training and freeze_batch_norm:
        _freeze_batch_norm_statistics(model)
    accumulator = BinaryMetricsAccumulator()
    totals = {"total": 0.0, "bce": 0.0, "dice": 0.0}
    sample_count = 0

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            losses = loss_function(logits, masks, valid_masks)
            if is_training:
                losses["total"].backward()
                optimizer.step()

            batch_size = images.shape[0]
            sample_count += batch_size
            for name in totals:
                totals[name] += float(losses[name].detach().cpu()) * batch_size
            accumulator.update(logits.detach(), masks, valid_masks)

    if sample_count == 0:
        raise RuntimeError("No batches were processed")
    metrics = accumulator.compute()
    return EpochResult(
        loss=totals["total"] / sample_count,
        bce=totals["bce"] / sample_count,
        dice_loss=totals["dice"] / sample_count,
        samples=sample_count,
        **metrics,
    )


def _make_loader(
    dataset: BinarySegmentationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=segmentation_collate,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _freeze_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: EpochResult,
    config: ExperimentConfig,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation": asdict(validation_metrics),
            "experiment_id": config.experiment_id,
            "input_size": config.data.input_size,
        },
        path,
    )


def _save_validation_preview(
    model: nn.Module,
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    output_directory: Path,
) -> None:
    batch = next(iter(validation_loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(images)
    save_prediction_preview(
        images[:4],
        masks[:4],
        logits[:4],
        output_directory / "validation_predictions.png",
        sample_ids=batch["sample_id"][:4],
    )


def _write_run_metadata(
    config: ExperimentConfig,
    output_directory: Path,
    device: torch.device,
    training: Any,
    pretrained: bool,
) -> None:
    metadata = {
        "experiment": _jsonable(asdict(config)),
        "effective_training": _jsonable(asdict(training)),
        "effective_pretrained": pretrained,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
        "process_id": os.getpid(),
    }
    (output_directory / "run.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
