from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from fm_to_edge_seg import __version__
from fm_to_edge_seg.data.manifest import validate_manifest


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
    raise ValueError(f"Unsupported command: {args.command}")
