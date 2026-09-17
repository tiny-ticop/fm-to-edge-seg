from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

REQUIRED_COLUMNS = {"sample_id", "image_path", "mask_path", "split"}
ALLOWED_SPLITS = {"train", "val", "test", "ood_test"}
ALLOWED_MASK_VALUES = {0, 1, 255}


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    sample_id: str
    message: str


@dataclass
class DatasetValidationReport:
    manifest_path: Path
    sample_count: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.level == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.level == "warning" for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def format(self) -> str:
        lines = [
            f"manifest: {self.manifest_path}",
            f"samples: {self.sample_count}",
            "splits: "
            + (
                ", ".join(f"{name}={count}" for name, count in sorted(self.split_counts.items()))
                or "none"
            ),
            f"errors: {self.error_count}",
            f"warnings: {self.warning_count}",
        ]
        lines.extend(f"[{issue.level}] {issue.sample_id}: {issue.message}" for issue in self.issues)
        return "\n".join(lines)


def load_manifest(manifest_path: Path, data_root: Path | None = None) -> list[ManifestRecord]:
    manifest_path = manifest_path.resolve()
    root = data_root.resolve() if data_root else manifest_path.parent

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest is missing required columns: {missing_text}")

        records = []
        for row_number, row in enumerate(reader, start=2):
            if not any(row.values()):
                continue
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"Manifest row {row_number} has an empty sample_id")
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    image_path=_resolve_data_path(root, row["image_path"]),
                    mask_path=_resolve_data_path(root, row["mask_path"]),
                    split=row["split"].strip(),
                    metadata={
                        key: value for key, value in row.items() if key not in REQUIRED_COLUMNS
                    },
                )
            )
    return records


def validate_manifest(
    manifest_path: Path,
    data_root: Path | None = None,
) -> DatasetValidationReport:
    report = DatasetValidationReport(manifest_path=manifest_path.resolve())
    try:
        records = load_manifest(manifest_path, data_root=data_root)
    except (OSError, ValueError) as error:
        report.issues.append(ValidationIssue("error", "manifest", str(error)))
        return report

    report.sample_count = len(records)
    report.split_counts.update(record.split for record in records)

    duplicates = {key for key, count in Counter(r.sample_id for r in records).items() if count > 1}
    for duplicate in sorted(duplicates):
        report.issues.append(ValidationIssue("error", duplicate, "duplicate sample_id"))

    if not records:
        report.issues.append(ValidationIssue("error", "manifest", "manifest contains no samples"))

    for record in records:
        _validate_record(record, report)
    return report


def _resolve_data_path(root: Path, value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else (root / path).resolve()


def _validate_record(record: ManifestRecord, report: DatasetValidationReport) -> None:
    if record.split not in ALLOWED_SPLITS:
        report.issues.append(
            ValidationIssue(
                "error",
                record.sample_id,
                f"unsupported split '{record.split}'; expected one of {sorted(ALLOWED_SPLITS)}",
            )
        )

    missing_paths = [path for path in (record.image_path, record.mask_path) if not path.is_file()]
    if missing_paths:
        for path in missing_paths:
            report.issues.append(
                ValidationIssue("error", record.sample_id, f"file does not exist: {path}")
            )
        return

    try:
        with Image.open(record.image_path) as image:
            image.load()
            image_size = image.size
            if image.mode not in {"RGB", "L"}:
                report.issues.append(
                    ValidationIssue(
                        "warning", record.sample_id, f"unusual image mode: {image.mode}"
                    )
                )

        with Image.open(record.mask_path) as mask:
            mask.load()
            mask_size = mask.size
            mask_array = np.asarray(mask)
    except (OSError, UnidentifiedImageError) as error:
        report.issues.append(
            ValidationIssue("error", record.sample_id, f"image decode failed: {error}")
        )
        return

    if image_size != mask_size:
        report.issues.append(
            ValidationIssue(
                "error",
                record.sample_id,
                f"size mismatch: image={image_size}, mask={mask_size}",
            )
        )

    if mask_array.ndim != 2:
        report.issues.append(
            ValidationIssue(
                "error",
                record.sample_id,
                f"mask must be single-channel, got shape={mask_array.shape}",
            )
        )
        return

    values = {int(value) for value in np.unique(mask_array)}
    unexpected = values - ALLOWED_MASK_VALUES
    if unexpected:
        report.issues.append(
            ValidationIssue(
                "error",
                record.sample_id,
                f"mask contains unsupported values: {sorted(unexpected)}",
            )
        )
    if 1 not in values:
        report.issues.append(ValidationIssue("warning", record.sample_id, "mask has no foreground"))
