from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from fm_to_edge_seg import __version__
from fm_to_edge_seg.data.deepcrack import prepare_deepcrack
from fm_to_edge_seg.data.manifest import validate_manifest
from fm_to_edge_seg.data.preview import create_dataset_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fm-edge-seg",
        description="Foundation-model to edge segmentation research utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Print the active environment.")
    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate image/mask pairs described by a dataset manifest.",
    )
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base directory for relative paths; defaults to the manifest directory.",
    )
    prepare_parser = subparsers.add_parser(
        "prepare-deepcrack",
        help="Convert the official DeepCrack layout into the canonical dataset format.",
    )
    prepare_parser.add_argument("source", type=Path, help="Directory containing train_img etc.")
    prepare_parser.add_argument("output", type=Path, help="Destination dataset directory.")
    prepare_parser.add_argument("--val-fraction", type=float, default=0.2)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files previously generated in the output directory.",
    )
    preview_parser = subparsers.add_parser(
        "preview-dataset",
        help="Create a contact sheet of images, masks, and overlays.",
    )
    preview_parser.add_argument("manifest", type=Path)
    preview_parser.add_argument("output", type=Path)
    preview_parser.add_argument("--split", default="train")
    preview_parser.add_argument("--limit", type=int, default=8)
    preview_parser.add_argument("--seed", type=int, default=42)
    overfit_parser = subparsers.add_parser(
        "overfit-batch",
        help="Overfit a few samples to verify the complete training path.",
    )
    overfit_parser.add_argument("manifest", type=Path)
    overfit_parser.add_argument("output", type=Path)
    overfit_parser.add_argument("--split", default="train")
    overfit_parser.add_argument("--width", type=int, default=272)
    overfit_parser.add_argument("--height", type=int, default=192)
    overfit_parser.add_argument("--samples", type=int, default=2)
    overfit_parser.add_argument("--steps", type=int, default=30)
    overfit_parser.add_argument("--learning-rate", type=float, default=3e-3)
    overfit_parser.add_argument("--seed", type=int, default=42)
    overfit_parser.add_argument("--pretrained", action="store_true")
    overfit_parser.add_argument("--device", default="cpu")
    return parser


def run_doctor() -> int:
    print(f"fm-to-edge-seg: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"executable: {sys.executable}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return run_doctor()
    if args.command == "validate-manifest":
        report = validate_manifest(args.manifest, data_root=args.data_root)
        print(report.format())
        return 0 if report.is_valid else 1
    if args.command == "prepare-deepcrack":
        result = prepare_deepcrack(
            source_root=args.source,
            output_root=args.output,
            val_fraction=args.val_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"manifest: {result.manifest_path}")
        print(
            "samples: "
            + ", ".join(f"{split}={count}" for split, count in sorted(result.split_counts.items()))
        )
        return 0
    if args.command == "preview-dataset":
        result = create_dataset_preview(
            manifest_path=args.manifest,
            output_path=args.output,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )
        print(f"preview: {result}")
        return 0
    if args.command == "overfit-batch":
        from fm_to_edge_seg.training.overfit import run_overfit_experiment

        result = run_overfit_experiment(
            manifest_path=args.manifest,
            output_directory=args.output,
            split=args.split,
            input_size=(args.width, args.height),
            sample_count=args.samples,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            pretrained=args.pretrained,
            device_name=args.device,
        )
        print(
            f"overfit: loss {result.initial_loss:.6f} -> {result.final_loss:.6f}, "
            f"dice {result.initial_dice:.4f} -> {result.final_dice:.4f}"
        )
        print(f"artifacts: {result.output_directory}")
        return 0 if result.final_loss < result.initial_loss else 2
    raise ValueError(f"Unsupported command: {args.command}")
