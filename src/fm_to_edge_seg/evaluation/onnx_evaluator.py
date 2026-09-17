from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset, segmentation_collate
from fm_to_edge_seg.evaluation.preview import save_prediction_preview
from fm_to_edge_seg.metrics import BinaryMetricsAccumulator
from fm_to_edge_seg.training.config import ExperimentConfig


@dataclass(frozen=True)
class OnnxEvaluationResult:
    split: str
    samples: int
    iou: float
    dice: float
    precision: float
    recall: float
    elapsed_seconds: float
    milliseconds_per_image: float
    model_size_megabytes: float
    model_sha256: str
    intra_op_threads: int
    onnxruntime_version: str
    providers: list[str]


def evaluate_onnx(
    config: ExperimentConfig,
    model_path: Path,
    output_directory: Path,
    split: str = "test",
    num_workers: int = 0,
    intra_op_threads: int = 1,
    max_samples: int | None = None,
) -> OnnxEvaluationResult:
    if intra_op_threads <= 0:
        raise ValueError("intra_op_threads must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    import onnxruntime as ort

    model_path = model_path.resolve()
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_info = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    _validate_input_shape(input_info.shape, config.data.input_size)

    dataset = BinarySegmentationDataset(
        config.data.manifest,
        split=split,
        input_size=config.data.input_size,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=segmentation_collate,
    )
    accumulator = BinaryMetricsAccumulator()
    preview_batch = None
    started_at = time.perf_counter()
    for batch in loader:
        images = batch["image"]
        logits_array = session.run(
            [output_name],
            {input_info.name: images.numpy().astype(np.float32, copy=False)},
        )[0]
        logits = torch.from_numpy(logits_array)
        accumulator.update(logits, batch["mask"], batch["valid_mask"])
        if preview_batch is None:
            preview_batch = (images, batch["mask"], logits, batch["sample_id"])
    elapsed = time.perf_counter() - started_at
    sample_count = len(dataset)
    metrics = accumulator.compute()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    result = OnnxEvaluationResult(
        split=split,
        samples=sample_count,
        elapsed_seconds=elapsed,
        milliseconds_per_image=elapsed * 1000.0 / sample_count,
        model_size_megabytes=model_path.stat().st_size / (1024**2),
        model_sha256=_sha256(model_path),
        intra_op_threads=intra_op_threads,
        onnxruntime_version=ort.__version__,
        providers=session.get_providers(),
        **metrics,
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    if preview_batch is not None:
        images, masks, logits, sample_ids = preview_batch
        save_prediction_preview(
            images,
            masks,
            logits,
            output_directory / "predictions.png",
            sample_ids=sample_ids,
        )
    return result


def _validate_input_shape(shape: list[int | str | None], input_size: tuple[int, int]) -> None:
    width, height = input_size
    expected = [1, 3, height, width]
    if list(shape) != expected:
        raise ValueError(f"ONNX input shape {shape} does not match config shape {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
