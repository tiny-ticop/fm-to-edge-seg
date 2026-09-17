from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset, segmentation_collate
from fm_to_edge_seg.evaluation.preview import save_prediction_preview
from fm_to_edge_seg.metrics import BinaryMetricsAccumulator
from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import ExperimentConfig


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    samples: int
    iou: float
    dice: float
    precision: float
    recall: float
    elapsed_seconds: float
    milliseconds_per_image: float
    parameters: int
    checkpoint_size_megabytes: float


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_directory: Path,
    split: str = "test",
    device_name: str = "auto",
    num_workers: int = 0,
) -> EvaluationResult:
    device = _select_device(device_name)
    dataset = BinarySegmentationDataset(
        config.data.manifest,
        split=split,
        input_size=config.data.input_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=segmentation_collate,
    )
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MobileNetV3LiteUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    accumulator = BinaryMetricsAccumulator()
    preview_batch = None
    started_at = time.perf_counter()
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
            accumulator.update(logits, masks, valid_masks)
            if preview_batch is None:
                preview_batch = (images, masks, logits, batch["sample_id"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at
    metrics = accumulator.compute()
    result = EvaluationResult(
        split=split,
        samples=len(dataset),
        elapsed_seconds=elapsed,
        milliseconds_per_image=elapsed * 1000 / len(dataset),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_size_megabytes=checkpoint_path.stat().st_size / (1024**2),
        **metrics,
    )

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    if preview_batch is not None:
        images, masks, logits, sample_ids = preview_batch
        save_prediction_preview(
            images[:4],
            masks[:4],
            logits[:4],
            output_directory / "predictions.png",
            sample_ids=sample_ids[:4],
        )
    return result


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
