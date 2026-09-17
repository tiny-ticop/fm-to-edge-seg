from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fm_to_edge_seg.models import MobileNetV3LiteUNet
from fm_to_edge_seg.training.config import ExperimentConfig


@dataclass(frozen=True)
class OnnxExportResult:
    model_path: str
    metadata_path: str
    checkpoint_path: str
    input_size: tuple[int, int]
    input_name: str
    output_name: str
    output_shape: tuple[int, ...]
    opset: int
    parameters: int
    model_size_megabytes: float
    model_sha256: str
    max_absolute_error: float
    mean_absolute_error: float
    torch_version: str
    onnx_version: str
    onnxruntime_version: str


@dataclass(frozen=True)
class OnnxBenchmarkResult:
    model_path: str
    model_sha256: str
    model_size_megabytes: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    warmup_runs: int
    measured_runs: int
    intra_op_threads: int
    mean_milliseconds: float
    median_milliseconds: float
    p90_milliseconds: float
    p95_milliseconds: float
    frames_per_second: float
    parameters: int
    onnxruntime_version: str
    providers: list[str]
    platform: str
    processor: str
    python_version: str


def export_student_onnx(
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    max_allowed_error: float = 1e-4,
) -> OnnxExportResult:
    if opset < 17:
        raise ValueError("opset must be 17 or newer")
    if max_allowed_error <= 0:
        raise ValueError("max_allowed_error must be positive")

    import onnx
    import onnxruntime as ort

    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("output_path must end with .onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MobileNetV3LiteUNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    width, height = config.data.input_size
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    example_input = torch.randn((1, 3, height, width), generator=generator)
    with torch.inference_mode():
        torch_output = model(example_input).cpu().numpy()

    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=True,
    )
    onnx_model = onnx.load(output_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_metadata = {
        "fm_to_edge_seg.parameters": str(parameter_count),
        "fm_to_edge_seg.input_width": str(width),
        "fm_to_edge_seg.input_height": str(height),
        "fm_to_edge_seg.image_mean": "0.485,0.456,0.406",
        "fm_to_edge_seg.image_std": "0.229,0.224,0.225",
        "fm_to_edge_seg.mask_threshold": "0.5",
    }
    for key, value in model_metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, output_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(["logits"], {"image": example_input.numpy()})[0]
    absolute_error = np.abs(torch_output - onnx_output)
    max_error = float(absolute_error.max())
    mean_error = float(absolute_error.mean())
    if max_error > max_allowed_error:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX parity check failed: max absolute error {max_error:.8f} "
            f"> allowed {max_allowed_error:.8f}"
        )

    metadata_path = output_path.with_suffix(".json")
    result = OnnxExportResult(
        model_path=str(output_path),
        metadata_path=str(metadata_path),
        checkpoint_path=str(checkpoint_path),
        input_size=(width, height),
        input_name="image",
        output_name="logits",
        output_shape=tuple(int(value) for value in onnx_output.shape),
        opset=opset,
        parameters=parameter_count,
        model_size_megabytes=output_path.stat().st_size / (1024**2),
        model_sha256=_sha256(output_path),
        max_absolute_error=max_error,
        mean_absolute_error=mean_error,
        torch_version=torch.__version__,
        onnx_version=onnx.__version__,
        onnxruntime_version=ort.__version__,
    )
    metadata_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def benchmark_onnx(
    model_path: Path,
    output_path: Path,
    warmup_runs: int = 10,
    measured_runs: int = 100,
    intra_op_threads: int = 1,
    seed: int = 42,
) -> OnnxBenchmarkResult:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")
    if intra_op_threads <= 0:
        raise ValueError("intra_op_threads must be positive")

    import onnx
    import onnxruntime as ort

    model_path = model_path.resolve()
    output_path = output_path.resolve()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    input_shape = _concrete_input_shape(input_info.shape)
    generator = np.random.default_rng(seed)
    input_array = generator.standard_normal(input_shape, dtype=np.float32)
    feeds = {input_info.name: input_array}

    for _ in range(warmup_runs):
        session.run(None, feeds)
    latencies = []
    output_arrays = []
    for _ in range(measured_runs):
        started_at = time.perf_counter_ns()
        output_arrays = session.run(None, feeds)
        elapsed = time.perf_counter_ns() - started_at
        latencies.append(elapsed / 1_000_000.0)

    mean_latency = statistics.fmean(latencies)
    onnx_model = onnx.load(model_path, load_external_data=False)
    result = OnnxBenchmarkResult(
        model_path=str(model_path),
        model_sha256=_sha256(model_path),
        model_size_megabytes=model_path.stat().st_size / (1024**2),
        input_shape=input_shape,
        output_shape=tuple(int(value) for value in output_arrays[0].shape),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        intra_op_threads=intra_op_threads,
        mean_milliseconds=mean_latency,
        median_milliseconds=statistics.median(latencies),
        p90_milliseconds=_percentile(latencies, 90),
        p95_milliseconds=_percentile(latencies, 95),
        frames_per_second=1000.0 / mean_latency,
        parameters=_count_onnx_parameters(onnx_model),
        onnxruntime_version=ort.__version__,
        providers=session.get_providers(),
        platform=platform.platform(),
        processor=platform.processor(),
        python_version=platform.python_version(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _concrete_input_shape(shape: list[int | str | None]) -> tuple[int, ...]:
    concrete = []
    for index, value in enumerate(shape):
        if isinstance(value, int) and value > 0:
            concrete.append(value)
        elif index == 0:
            concrete.append(1)
        else:
            raise ValueError(f"ONNX model has a dynamic non-batch input dimension: {shape}")
    return tuple(concrete)


def _count_onnx_parameters(model: object) -> int:
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if "fm_to_edge_seg.parameters" in metadata:
        return int(metadata["fm_to_edge_seg.parameters"])
    return sum(
        int(np.prod(initializer.dims, dtype=np.int64))
        for initializer in model.graph.initializer
    )


def _percentile(values: list[float], percentile: int) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
