from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from fm_to_edge_seg import __version__
from fm_to_edge_seg.data.binary_dataset import prepare_binary_dataset
from fm_to_edge_seg.data.deepcrack import prepare_deepcrack
from fm_to_edge_seg.data.manifest import validate_manifest
from fm_to_edge_seg.data.preview import create_dataset_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fm-edge-seg",
        description="Foundation-model to edge segmentation research utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Print Python, PyTorch, and GPU diagnostics.")
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
    binary_parser = subparsers.add_parser(
        "prepare-binary-dataset",
        help="Prepare matching images/ and masks/ folders as a canonical dataset.",
    )
    binary_parser.add_argument("source", type=Path)
    binary_parser.add_argument("output", type=Path)
    binary_parser.add_argument("--val-fraction", type=float, default=0.2)
    binary_parser.add_argument("--test-fraction", type=float, default=0.0)
    binary_parser.add_argument("--seed", type=int, default=42)
    binary_parser.add_argument("--metadata", type=Path, default=None)
    binary_parser.add_argument("--overwrite", action="store_true")
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
    train_parser = subparsers.add_parser(
        "train",
        help="Train and validate an experiment defined by YAML configuration.",
    )
    train_parser.add_argument("config", type=Path)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--num-workers", type=int, default=None)
    train_parser.add_argument("--max-train-batches", type=int, default=None)
    train_parser.add_argument("--max-validation-batches", type=int, default=None)
    train_parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained encoder weights for pipeline smoke tests.",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained checkpoint on a manifest split.",
    )
    evaluate_parser.add_argument("config", type=Path)
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("output", type=Path)
    evaluate_parser.add_argument("--split", default="test")
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    cache_parser = subparsers.add_parser(
        "create-reference-teacher-cache",
        help="Create a label-derived teacher cache to verify the distillation pipeline.",
    )
    cache_parser.add_argument("manifest", type=Path)
    cache_parser.add_argument("output", type=Path)
    cache_parser.add_argument("--split", default="train")
    cache_parser.add_argument("--foreground-probability", type=float, default=0.99)
    cache_parser.add_argument("--overwrite", action="store_true")
    sam3_parser = subparsers.add_parser(
        "create-sam3-teacher-cache",
        help="Run official SAM 3 with a text prompt and create a teacher cache.",
    )
    sam3_parser.add_argument("manifest", type=Path)
    sam3_parser.add_argument("output", type=Path)
    sam3_parser.add_argument("--prompt", required=True)
    sam3_parser.add_argument("--split", default="train")
    sam3_parser.add_argument("--device", default="cuda")
    sam3_parser.add_argument("--score-threshold", type=float, default=0.5)
    sam3_parser.add_argument("--foreground-probability", type=float, default=0.99)
    sam3_parser.add_argument("--background-confidence", type=float, default=0.25)
    sam3_parser.add_argument("--checkpoint", type=Path, default=None)
    sam3_parser.add_argument("--max-samples", type=int, default=None)
    sam3_parser.add_argument("--overwrite", action="store_true")
    teacher_preview_parser = subparsers.add_parser(
        "preview-teacher-cache",
        help="Compare ground truth, cached teacher masks, and confidence.",
    )
    teacher_preview_parser.add_argument("manifest", type=Path)
    teacher_preview_parser.add_argument("cache", type=Path)
    teacher_preview_parser.add_argument("output", type=Path)
    teacher_preview_parser.add_argument("--split", default="train")
    teacher_preview_parser.add_argument("--limit", type=int, default=8)
    teacher_preview_parser.add_argument("--seed", type=int, default=42)
    teacher_evaluation_parser = subparsers.add_parser(
        "evaluate-teacher-cache",
        help="Evaluate cached teacher masks against ground truth.",
    )
    teacher_evaluation_parser.add_argument("manifest", type=Path)
    teacher_evaluation_parser.add_argument("cache", type=Path)
    teacher_evaluation_parser.add_argument("output", type=Path)
    teacher_evaluation_parser.add_argument("--split", default="train")
    teacher_evaluation_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser


def run_doctor() -> int:
    print(f"fm-to-edge-seg: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"executable: {sys.executable}")
    try:
        import torch
        import torchvision

        print(f"torch: {torch.__version__}")
        print(f"torchvision: {torchvision.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"torch_cuda_runtime: {torch.version.cuda}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                memory_gib = properties.total_memory / (1024**3)
                print(f"gpu[{index}]: {properties.name} ({memory_gib:.1f} GiB)")
        else:
            print("gpu: not available (CPU training remains available)")
    except ImportError as error:
        print(f"pytorch: not installed ({error})")
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
    if args.command == "prepare-binary-dataset":
        result = prepare_binary_dataset(
            source_root=args.source,
            output_root=args.output,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            metadata_path=args.metadata,
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
    if args.command == "train":
        from fm_to_edge_seg.training.config import load_experiment_config
        from fm_to_edge_seg.training.trainer import train_experiment

        config = load_experiment_config(args.config)
        result = train_experiment(
            config,
            device_override=args.device,
            epochs_override=args.epochs,
            num_workers_override=args.num_workers,
            pretrained_override=False if args.no_pretrained else None,
            max_train_batches=args.max_train_batches,
            max_validation_batches=args.max_validation_batches,
        )
        print(
            f"training_complete: best_epoch={result.best_epoch}, "
            f"best_validation_dice={result.best_validation_dice:.4f}"
        )
        print(f"artifacts: {result.output_directory}")
        return 0
    if args.command == "evaluate":
        from fm_to_edge_seg.evaluation.evaluator import evaluate_checkpoint
        from fm_to_edge_seg.training.config import load_experiment_config

        config = load_experiment_config(args.config)
        result = evaluate_checkpoint(
            config=config,
            checkpoint_path=args.checkpoint,
            output_directory=args.output,
            split=args.split,
            device_name=args.device,
            num_workers=args.num_workers,
        )
        print(
            f"evaluation_complete: split={result.split}, samples={result.samples}, "
            f"dice={result.dice:.4f}, iou={result.iou:.4f}, "
            f"milliseconds_per_image={result.milliseconds_per_image:.2f}"
        )
        print(f"artifacts: {args.output.resolve()}")
        return 0
    if args.command == "create-reference-teacher-cache":
        from fm_to_edge_seg.distillation import create_reference_teacher_cache

        output = create_reference_teacher_cache(
            manifest_path=args.manifest,
            output_root=args.output,
            split=args.split,
            foreground_probability=args.foreground_probability,
            overwrite=args.overwrite,
        )
        print(f"teacher_cache: {output}")
        print("teacher_kind: reference_ground_truth_mask (pipeline verification only)")
        return 0
    if args.command == "create-sam3-teacher-cache":
        from fm_to_edge_seg.distillation.sam3_adapter import (
            Sam3TextTeacher,
            create_sam3_teacher_cache,
        )

        teacher = Sam3TextTeacher(
            prompt=args.prompt,
            device=args.device,
            score_threshold=args.score_threshold,
            foreground_probability=args.foreground_probability,
            background_confidence=args.background_confidence,
            checkpoint_path=args.checkpoint,
        )
        output = create_sam3_teacher_cache(
            manifest_path=args.manifest,
            output_root=args.output,
            teacher=teacher,
            split=args.split,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
        )
        print(f"teacher_cache: {output}")
        print(f"teacher_kind: sam3_text_prompt, prompt={args.prompt!r}")
        return 0
    if args.command == "preview-teacher-cache":
        from fm_to_edge_seg.distillation.preview import create_teacher_cache_preview

        output = create_teacher_cache_preview(
            manifest_path=args.manifest,
            cache_root=args.cache,
            output_path=args.output,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )
        print(f"teacher_preview: {output}")
        return 0
    if args.command == "evaluate-teacher-cache":
        from fm_to_edge_seg.distillation.evaluation import evaluate_teacher_cache

        result = evaluate_teacher_cache(
            manifest_path=args.manifest,
            cache_root=args.cache,
            output_directory=args.output,
            split=args.split,
            confidence_threshold=args.confidence_threshold,
        )
        print(
            f"teacher_evaluation: samples={result.evaluated_samples}/{result.manifest_samples}, "
            f"dice={result.dice:.4f}, iou={result.iou:.4f}, "
            f"confident_pixels={result.confident_pixel_fraction:.4f}"
        )
        print(f"artifacts: {args.output.resolve()}")
        return 0
    raise ValueError(f"Unsupported command: {args.command}")
