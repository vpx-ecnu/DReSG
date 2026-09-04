#!/usr/bin/env python3
"""Regenerate train-view renders or video for a saved DReSG run."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dresg.inference import render_run_train_views, render_run_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_views = commands.add_parser("train-views")
    train_views.add_argument("--run-dir", required=True, type=Path)
    train_views.add_argument("--device", type=torch.device)

    video = commands.add_parser("video")
    video.add_argument("--run-dir", required=True, type=Path)
    video.add_argument("--path", type=Path)
    video.add_argument("--device", type=torch.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train-views":
        metrics = render_run_train_views(args.run_dir, device=args.device)
        print(metrics)
    elif args.command == "video":
        output_path = render_run_video(
            args.run_dir,
            path=args.path,
            device=args.device,
        )
        print({"output": str(output_path)})
    else:
        raise AssertionError(f"Unhandled render command: {args.command}")


if __name__ == "__main__":
    main()
