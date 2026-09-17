from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fm_to_edge_seg.data.manifest import load_manifest
from fm_to_edge_seg.distillation.cache import TeacherCache


@dataclass(frozen=True)
class TeacherEvaluationResult:
    split: str
    evaluated_samples: int
    manifest_samples: int
    iou: float
    dice: float
    precision: float
    recall: float
    confident_pixel_fraction: float


def evaluate_teacher_cache(
    manifest_path: Path,
    cache_root: Path,
    output_directory: Path,
    split: str = "train",
    confidence_threshold: float = 0.5,
) -> TeacherEvaluationResult:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"Manifest contains no samples for split '{split}'")
    cache = TeacherCache(cache_root)
    rows: list[dict[str, object]] = []
    total_tp = total_fp = total_fn = 0
    confident_pixels = valid_pixels = 0
    for record in records:
        if not cache.has_sample(record.sample_id):
            continue
        teacher = cache.load(record.sample_id)
        with Image.open(record.mask_path) as source:
            ground_truth = np.asarray(source.convert("L"), dtype=np.uint8)
        if teacher.logits.shape != ground_truth.shape:
            raise ValueError(f"Teacher/mask size mismatch for '{record.sample_id}'")
        valid = ground_truth != 255
        target = ground_truth == 1
        prediction = teacher.logits >= 0
        tp = int((prediction & target & valid).sum())
        fp = int((prediction & ~target & valid).sum())
        fn = int((~prediction & target & valid).sum())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        valid_count = int(valid.sum())
        confident_count = int(((teacher.confidence >= confidence_threshold) & valid).sum())
        valid_pixels += valid_count
        confident_pixels += confident_count
        rows.append(
            {
                "sample_id": record.sample_id,
                "iou": _ratio(tp, tp + fp + fn),
                "dice": _ratio(2 * tp, 2 * tp + fp + fn),
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
                "confident_pixel_fraction": _ratio(confident_count, valid_count),
            }
        )
    if not rows:
        raise ValueError(f"Teacher cache contains no samples for split '{split}'")

    result = TeacherEvaluationResult(
        split=split,
        evaluated_samples=len(rows),
        manifest_samples=len(records),
        iou=_ratio(total_tp, total_tp + total_fp + total_fn),
        dice=_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        precision=_ratio(total_tp, total_tp + total_fp),
        recall=_ratio(total_tp, total_tp + total_fn),
        confident_pixel_fraction=_ratio(confident_pixels, valid_pixels),
    )
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    with (output_directory / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
