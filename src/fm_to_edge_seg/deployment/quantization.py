from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Subset

from fm_to_edge_seg.data.dataset import BinarySegmentationDataset
from fm_to_edge_seg.training.config import ExperimentConfig


@dataclass(frozen=True)
class QuantizationResult:
    fp32_model_path: str
    int8_model_path: str
    metadata_path: str
    quant_format: str
    activation_type: str
    weight_type: str
    calibration_method: str
    per_channel: bool
    calibration_split: str
    calibration_samples: int
    calibration_sample_ids: list[str]
    manifest_sha256: str
    fp32_size_megabytes: float
    int8_size_megabytes: float
    size_reduction_percent: float
    fp32_sha256: str
    int8_sha256: str
    parity_max_absolute_error: float
    parity_mean_absolute_error: float
    onnxruntime_version: str


class SegmentationCalibrationDataReader:
    def __init__(
        self,
        dataset: BinarySegmentationDataset,
        sample_count: int,
        seed: int,
        input_name: str,
    ) -> None:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        selected_indices = sorted(indices[: min(sample_count, len(indices))])
        self.dataset = Subset(dataset, selected_indices)
        self.input_name = input_name
        self.sample_ids = [dataset.records[index].sample_id for index in selected_indices]
        self._iterator: Any = None
        self.rewind()

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            sample = next(self._iterator)
        except StopIteration:
            return None
        image = sample["image"].unsqueeze(0).numpy().astype(np.float32, copy=False)
        return {self.input_name: image}

    def rewind(self) -> None:
        self._iterator = iter(self.dataset)


def quantize_onnx_static(
    config: ExperimentConfig,
    fp32_model_path: Path,
    int8_model_path: Path,
    calibration_split: str = "train",
    calibration_samples: int = 32,
    calibration_method: str = "minmax",
    per_channel: bool = True,
) -> QuantizationResult:
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    if calibration_method not in {"minmax", "entropy", "percentile"}:
        raise ValueError("calibration_method must be minmax, entropy, or percentile")

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    fp32_model_path = fp32_model_path.resolve()
    int8_model_path = int8_model_path.resolve()
    if int8_model_path.suffix.lower() != ".onnx":
        raise ValueError("int8_model_path must end with .onnx")
    if fp32_model_path == int8_model_path:
        raise ValueError("INT8 output must differ from the FP32 input")
    int8_model_path.parent.mkdir(parents=True, exist_ok=True)

    probe_session = ort.InferenceSession(
        str(fp32_model_path), providers=["CPUExecutionProvider"]
    )
    input_name = probe_session.get_inputs()[0].name
    dataset = BinarySegmentationDataset(
        config.data.manifest,
        split=calibration_split,
        input_size=config.data.input_size,
    )
    reader = SegmentationCalibrationDataReader(
        dataset=dataset,
        sample_count=calibration_samples,
        seed=config.seed,
        input_name=input_name,
    )
    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    quantize_static(
        model_input=str(fp32_model_path),
        model_output=str(int8_model_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method_map[calibration_method],
        per_channel=per_channel,
        op_types_to_quantize=["Conv"],
    )

    int8_model = onnx.load(int8_model_path)
    onnx.checker.check_model(int8_model)
    metadata_values = {
        "fm_to_edge_seg.quantization": "static_int8_s8s8_qdq",
        "fm_to_edge_seg.calibration_method": calibration_method,
        "fm_to_edge_seg.calibration_samples": str(len(reader.sample_ids)),
        "fm_to_edge_seg.per_channel": str(per_channel).lower(),
    }
    existing_keys = {entry.key for entry in int8_model.metadata_props}
    for key, value in metadata_values.items():
        if key in existing_keys:
            continue
        entry = int8_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(int8_model, int8_model_path)

    reader.rewind()
    parity_input = reader.get_next()
    if parity_input is None:
        raise RuntimeError("Calibration dataset produced no samples")
    int8_session = ort.InferenceSession(
        str(int8_model_path), providers=["CPUExecutionProvider"]
    )
    fp32_output = probe_session.run(None, parity_input)[0]
    int8_output = int8_session.run(None, parity_input)[0]
    absolute_error = np.abs(fp32_output - int8_output)

    metadata_path = int8_model_path.with_suffix(".json")
    fp32_size = fp32_model_path.stat().st_size / (1024**2)
    int8_size = int8_model_path.stat().st_size / (1024**2)
    result = QuantizationResult(
        fp32_model_path=str(fp32_model_path),
        int8_model_path=str(int8_model_path),
        metadata_path=str(metadata_path),
        quant_format="QDQ",
        activation_type="QInt8",
        weight_type="QInt8",
        calibration_method=calibration_method,
        per_channel=per_channel,
        calibration_split=calibration_split,
        calibration_samples=len(reader.sample_ids),
        calibration_sample_ids=reader.sample_ids,
        manifest_sha256=_sha256(config.data.manifest),
        fp32_size_megabytes=fp32_size,
        int8_size_megabytes=int8_size,
        size_reduction_percent=(1.0 - int8_size / fp32_size) * 100.0,
        fp32_sha256=_sha256(fp32_model_path),
        int8_sha256=_sha256(int8_model_path),
        parity_max_absolute_error=float(absolute_error.max()),
        parity_mean_absolute_error=float(absolute_error.mean()),
        onnxruntime_version=ort.__version__,
    )
    metadata_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
