from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Subset

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset, segmentation_collate
from fm_to_edge_seg.evaluation.preview import save_prediction_preview
from fm_to_edge_seg.metrics import BinaryMetricsAccumulator
from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import ExperimentConfig


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    kind: str
    value: float | None = None


@dataclass(frozen=True)
class RobustnessConditionResult:
    condition: str
    parameters: dict[str, float]
    samples: int
    iou: float
    dice: float
    precision: float
    recall: float
    dice_drop_from_clean: float
    milliseconds_per_image: float


@dataclass(frozen=True)
class RobustnessResult:
    split: str
    samples_per_condition: int
    clean_dice: float
    mean_corrupted_dice: float
    mean_dice_drop: float
    worst_condition: str
    worst_dice: float
    conditions: list[RobustnessConditionResult]


DEFAULT_CORRUPTIONS = (
    CorruptionSpec("clean", "clean"),
    CorruptionSpec("brightness_dark", "brightness", 0.6),
    CorruptionSpec("brightness_bright", "brightness", 1.4),
    CorruptionSpec("contrast_low", "contrast", 0.6),
    CorruptionSpec("color_desaturated", "saturation", 0.4),
    CorruptionSpec("gaussian_noise", "gaussian_noise", 0.08),
    CorruptionSpec("gaussian_blur", "gaussian_blur", 1.5),
)
CORRUPTION_NAMES = tuple(spec.name for spec in DEFAULT_CORRUPTIONS)


class DeterministicImageCorruption:
    """Apply a reproducible image-only corruption while preserving the mask."""

    def __init__(self, spec: CorruptionSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        value = self.spec.value
        if self.spec.kind == "clean":
            output = image.copy()
        elif self.spec.kind == "brightness" and value is not None:
            output = ImageEnhance.Brightness(image).enhance(value)
        elif self.spec.kind == "contrast" and value is not None:
            output = ImageEnhance.Contrast(image).enhance(value)
        elif self.spec.kind == "saturation" and value is not None:
            output = ImageEnhance.Color(image).enhance(value)
        elif self.spec.kind == "gaussian_blur" and value is not None:
            output = image.filter(ImageFilter.GaussianBlur(radius=value))
        elif self.spec.kind == "gaussian_noise" and value is not None:
            output = self._add_gaussian_noise(image, value)
        else:
            raise ValueError(f"Unsupported corruption: {self.spec}")
        return output, mask

    def _add_gaussian_noise(self, image: Image.Image, standard_deviation: float) -> Image.Image:
        image_array = np.asarray(image, dtype=np.uint8)
        digest = hashlib.sha256()
        digest.update(image_array.tobytes())
        digest.update(self.spec.name.encode("utf-8"))
        digest.update(str(self.seed).encode("ascii"))
        sample_seed = int.from_bytes(digest.digest()[:8], "little")
        generator = np.random.default_rng(sample_seed)
        noise = generator.normal(0.0, standard_deviation * 255.0, image_array.shape)
        corrupted = np.clip(image_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(corrupted, mode="RGB")


def evaluate_robustness(
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_directory: Path,
    split: str = "test",
    device_name: str = "auto",
    num_workers: int = 0,
    condition_names: list[str] | None = None,
    max_samples: int | None = None,
) -> RobustnessResult:
    specs = _select_corruptions(condition_names)
    if specs[0].name != "clean":
        specs = [next(spec for spec in DEFAULT_CORRUPTIONS if spec.name == "clean"), *specs]
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    device = _select_device(device_name)
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MobileNetV3LiteUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_results = [
        _evaluate_condition(
            config=config,
            model=model,
            spec=spec,
            split=split,
            device=device,
            num_workers=num_workers,
            output_directory=output_directory,
            max_samples=max_samples,
        )
        for spec in specs
    ]
    clean_dice = raw_results[0].dice
    results = [
        RobustnessConditionResult(
            **{
                **asdict(result),
                "dice_drop_from_clean": clean_dice - result.dice,
            }
        )
        for result in raw_results
    ]
    corrupted = [result for result in results if result.condition != "clean"]
    comparison_pool = corrupted or results
    worst = min(comparison_pool, key=lambda result: result.dice)
    mean_corrupted_dice = sum(result.dice for result in comparison_pool) / len(comparison_pool)
    summary = RobustnessResult(
        split=split,
        samples_per_condition=results[0].samples,
        clean_dice=clean_dice,
        mean_corrupted_dice=mean_corrupted_dice,
        mean_dice_drop=clean_dice - mean_corrupted_dice,
        worst_condition=worst.condition,
        worst_dice=worst.dice,
        conditions=results,
    )
    (output_directory / "robustness.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    _write_csv(output_directory / "robustness.csv", results)
    return summary


def _evaluate_condition(
    config: ExperimentConfig,
    model: MobileNetV3LiteUNet,
    spec: CorruptionSpec,
    split: str,
    device: torch.device,
    num_workers: int,
    output_directory: Path,
    max_samples: int | None,
) -> RobustnessConditionResult:
    dataset = BinarySegmentationDataset(
        config.data.manifest,
        split=split,
        input_size=config.data.input_size,
        joint_transform=DeterministicImageCorruption(spec, config.seed),
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=segmentation_collate,
    )
    accumulator = BinaryMetricsAccumulator()
    preview_batch = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
    sample_count = len(dataset)
    metrics = accumulator.compute()
    if preview_batch is not None:
        images, masks, logits, sample_ids = preview_batch
        save_prediction_preview(
            images[:4],
            masks[:4],
            logits[:4],
            output_directory / "previews" / f"{spec.name}.png",
            sample_ids=sample_ids[:4],
        )
    parameters = {} if spec.value is None else {spec.kind: spec.value}
    return RobustnessConditionResult(
        condition=spec.name,
        parameters=parameters,
        samples=sample_count,
        dice_drop_from_clean=0.0,
        milliseconds_per_image=elapsed * 1000.0 / sample_count,
        **metrics,
    )


def _select_corruptions(condition_names: list[str] | None) -> list[CorruptionSpec]:
    if condition_names is None:
        return list(DEFAULT_CORRUPTIONS)
    if not condition_names:
        raise ValueError("At least one robustness condition is required")
    unknown = sorted(set(condition_names) - set(CORRUPTION_NAMES))
    if unknown:
        raise ValueError(f"Unknown robustness conditions: {unknown}")
    requested = set(condition_names)
    return [spec for spec in DEFAULT_CORRUPTIONS if spec.name in requested]


def _write_csv(path: Path, results: list[RobustnessConditionResult]) -> None:
    fieldnames = [
        "condition",
        "samples",
        "iou",
        "dice",
        "precision",
        "recall",
        "dice_drop_from_clean",
        "milliseconds_per_image",
        "parameters",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["parameters"] = json.dumps(row["parameters"], sort_keys=True)
            writer.writerow(row)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
