#!/usr/bin/env python3
"""Build a reusable scene-bound video camera path artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from dresg.data.colmap import load_colmap_scene
from dresg.inference.paths import (
    VideoPathRequest,
    build_video_path,
    save_video_path,
)


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not 0.0 < number < float("inf"):
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--factor", required=True, type=_positive_integer)
    parser.add_argument(
        "--camera-source",
        choices=["all", "train_split", "interior"],
        required=True,
    )
    parser.add_argument(
        "--test-every",
        type=_positive_integer,
        help="required only when --camera-source=train_split",
    )
    parser.add_argument("--n-frames", required=True, type=_positive_integer)
    parser.add_argument("--output", required=True, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="trajectory", required=True)

    llff = commands.add_parser("llff")
    _add_common_arguments(llff)
    llff.add_argument(
        "--coord-mode",
        choices=["none", "flip_y", "flip_z", "flip_yz"],
        required=True,
    )
    llff.add_argument(
        "--radius-scale",
        required=True,
        type=_positive_float,
    )
    llff.add_argument(
        "--centered-radius",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    builtin = commands.add_parser("builtin-spiral")
    _add_common_arguments(builtin)
    builtin.add_argument(
        "--radius-scale",
        required=True,
        type=_positive_float,
    )

    tnt = commands.add_parser("tnt")
    _add_common_arguments(tnt)
    tnt.add_argument(
        "--ellipse-scale",
        required=True,
        type=_positive_float,
    )

    ellipse = commands.add_parser("ellipse")
    _add_common_arguments(ellipse)

    interpolated = commands.add_parser("interpolated")
    _add_common_arguments(interpolated)
    return parser.parse_args()


def _request(args: argparse.Namespace) -> VideoPathRequest:
    trajectory_names = {
        "llff": "llff_spiral",
        "builtin-spiral": "builtin_spiral",
        "tnt": "tnt_ellipse",
        "ellipse": "ellipse_z",
        "interpolated": "interpolated",
    }
    return VideoPathRequest(
        trajectory=trajectory_names[args.trajectory],
        camera_source=args.camera_source,
        test_every=args.test_every,
        n_frames=args.n_frames,
        coord_mode=getattr(args, "coord_mode", None),
        llff_radius_scale=getattr(args, "radius_scale", None),
        ellipse_scale=getattr(args, "ellipse_scale", None),
        centered_llff_radius=getattr(args, "centered_radius", None),
    )


def main() -> None:
    args = parse_args()
    source = load_colmap_scene(
        scene_dir=args.scene_dir,
        factor=args.factor,
    )
    path = build_video_path(source, _request(args))
    save_video_path(args.output, path)
    print(
        {
            "output": str(args.output),
            "frames": path.frame_count,
            "width": path.width,
            "height": path.height,
            "scene_fingerprint": path.scene_fingerprint,
        }
    )


if __name__ == "__main__":
    main()
