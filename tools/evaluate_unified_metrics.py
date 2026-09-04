#!/usr/bin/env python3
"""CLI wrapper for DReSG paper metrics.

Core implementation lives in :mod:`dresg.evaluation.paper_metrics`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dresg.evaluation.features import DEFAULT_CLIP_MODEL, DEFAULT_DINO_MODEL
from dresg.evaluation.layout import DEFAULT_METHODS, parse_key_path, parse_methods
from dresg.evaluation.paper_metrics import evaluate_paper_metrics, write_metric_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qual-root", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--style", required=True, help="Style key, e.g. 017.")
    parser.add_argument("--style-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument(
        "--offline-models",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--consistency-samples", type=int, default=6)
    parser.add_argument("--short-gap", type=str, default="1")
    parser.add_argument("--long-gap", type=str, default="n/2")
    parser.add_argument("--view-mode", choices=["all", "rendered"], default="all")
    parser.add_argument(
        "--result-dir", action="append", help="Optional method=result_dir containing renders and video."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output_dir}")
    methods = parse_methods(args.methods)
    style = args.style
    result_dirs = parse_key_path(args.result_dir)
    rows = evaluate_paper_metrics(
        qual_root=args.qual_root,
        scene_dir=args.scene_dir,
        scene=args.scene,
        style=style,
        style_path=args.style_path,
        methods=methods,
        factor=args.factor,
        device=args.device,
        batch_size=args.batch_size,
        consistency_samples=args.consistency_samples,
        short_gap=args.short_gap,
        long_gap=args.long_gap,
        view_mode=args.view_mode,
        result_dirs=result_dirs,
        clip_model=args.clip_model,
        dino_model=args.dino_model,
        offline_models=args.offline_models,
    )

    config = {
        "scene": args.scene,
        "style": style,
        "methods": methods,
        "factor": args.factor,
        "short_gap": args.short_gap,
        "long_gap": args.long_gap,
        "consistency_samples": args.consistency_samples,
        "view_mode": args.view_mode,
        "device": str(args.device),
        "batch_size": args.batch_size,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "offline_models": args.offline_models,
        "result_dirs": {method: str(path) for method, path in result_dirs.items()},
    }
    write_metric_bundle(args.output_dir, rows, config)
    print(args.output_dir / "paper_metrics.csv")


if __name__ == "__main__":
    main()
