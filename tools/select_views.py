#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dresg.utils.parsing import parse_int_list
from dresg.utils.view_selection.workflow import (
    ViewSelectionRequest,
    run_view_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select active views by depth-gated per-Gaussian support.",
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--base-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--candidate-views", default="")
    parser.add_argument("--seed-views", default="")
    parser.add_argument("--pool-grid-size", type=int, default=3)
    parser.add_argument("--pool-radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--depth-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-tolerance-ratio", type=float, default=0.01)
    parser.add_argument("--min-weight", type=float, default=1e-4)
    parser.add_argument("--target-fraction-of-max", type=float, default=0.98)
    parser.add_argument("--min-marginal-gain-ratio", type=float, default=0.001)
    parser.add_argument("--stop-coverage-ratio", type=float, default=0.9999)
    parser.add_argument("--max-select", type=int)
    return parser.parse_args()


def _optional_int_list(value: str) -> tuple[int, ...]:
    return tuple(parse_int_list(value)) if value.strip() else ()


def main() -> None:
    args = parse_args()
    result = run_view_selection(
        ViewSelectionRequest(
            scene_dir=args.scene_dir,
            base_ply=args.base_ply,
            output_dir=args.output_dir,
            dataset=args.dataset,
            scene=args.scene,
            device=args.device,
            factor=args.factor,
            render_scale=args.render_scale,
            candidate_views=_optional_int_list(args.candidate_views),
            seed_views=_optional_int_list(args.seed_views),
            pool_grid_size=args.pool_grid_size,
            pool_radius_scale=args.pool_radius_scale,
            depth_gate=args.depth_gate,
            depth_tolerance=args.depth_tolerance,
            depth_tolerance_ratio=args.depth_tolerance_ratio,
            min_weight=args.min_weight,
            target_fraction_of_max=args.target_fraction_of_max,
            min_marginal_gain_ratio=args.min_marginal_gain_ratio,
            stop_coverage_ratio=args.stop_coverage_ratio,
            max_select=args.max_select,
        )
    )
    print(
        {
            "output_dir": str(args.output_dir),
            "selected_views": result["selected_views"],
        }
    )


if __name__ == "__main__":
    main()
