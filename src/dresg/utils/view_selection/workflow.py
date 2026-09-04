"""End-to-end active-view selection workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from dresg.data.cameras import build_scaled_cameras
from dresg.data.colmap import load_colmap_scene
from dresg.models.gs import build_gaussian_scene
from dresg.utils.json_io import save_json
from dresg.utils.view_selection.greedy import greedy_select_views
from dresg.utils.view_selection.support import compute_view_support


@dataclass(frozen=True)
class ViewSelectionRequest:
    scene_dir: Path
    base_ply: Path
    output_dir: Path
    dataset: str
    scene: str
    device: torch.device
    factor: int
    render_scale: float
    candidate_views: tuple[int, ...]
    seed_views: tuple[int, ...]
    pool_grid_size: int
    pool_radius_scale: float
    depth_gate: bool
    depth_tolerance: float
    depth_tolerance_ratio: float
    min_weight: float
    target_fraction_of_max: float
    min_marginal_gain_ratio: float
    stop_coverage_ratio: float
    max_select: int | None


def run_view_selection(request: ViewSelectionRequest) -> dict[str, object]:
    device = request.device
    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_colmap_scene(
        scene_dir=request.scene_dir,
        factor=request.factor,
    )
    candidate_views = list(request.candidate_views)
    if not candidate_views:
        candidate_views = list(range(len(source)))
    seed_views = list(request.seed_views)
    cameras = build_scaled_cameras(
        source=source,
        view_ids=candidate_views,
        device=device,
        render_scale=request.render_scale,
        label="view selection",
    )
    scene = build_gaussian_scene(
        ply_path=request.base_ply,
        device=device,
        optimize_geometry=False,
        optimize_quats=False,
        max_mean_delta=0.0,
        max_scale_delta=0.0,
        max_quat_delta=0.0,
    )

    supports = []
    view_metrics = []
    for camera in cameras:
        support = compute_view_support(
            scene=scene,
            camera=camera,
            pool_grid_size=request.pool_grid_size,
            pool_radius_scale=request.pool_radius_scale,
            depth_gate=request.depth_gate,
            depth_tolerance=request.depth_tolerance,
            depth_tolerance_ratio=request.depth_tolerance_ratio,
        )
        supports.append(support)
        view_metrics.append(
            {
                "view_index": support.view_index,
                "supported_gaussians": support.gaussian_indices.numel(),
                "visible_samples": support.visible_samples,
                "depth_rejected": support.depth_rejected,
                "support_sum": support.values.sum().item(),
            }
        )

    curve = greedy_select_views(
        supports=supports,
        gaussian_count=scene.means().shape[0],
        seed_views=seed_views,
        min_weight=request.min_weight,
        target_fraction_of_max=request.target_fraction_of_max,
        max_select=request.max_select,
        stop_coverage_ratio=request.stop_coverage_ratio,
        min_marginal_gain_ratio=request.min_marginal_gain_ratio,
    )
    final = curve[-1]
    selected_views = list(final["views"])
    result = {
        "dataset": request.dataset,
        "scene": request.scene,
        "candidate_views": candidate_views,
        "selected_views": selected_views,
        "parameters": {
            "target_fraction_of_max": request.target_fraction_of_max,
            "min_weight": request.min_weight,
            "min_marginal_gain_ratio": request.min_marginal_gain_ratio,
            "stop_coverage_ratio": request.stop_coverage_ratio,
            "max_select": request.max_select,
            "depth_gate": request.depth_gate,
            "depth_tolerance": request.depth_tolerance,
            "depth_tolerance_ratio": request.depth_tolerance_ratio,
            "pool_grid_size": request.pool_grid_size,
            "pool_radius_scale": request.pool_radius_scale,
        },
        "view_metrics": view_metrics,
        "greedy_curve": curve,
    }
    save_json(output_dir / "coverage_selection.json", result)

    config = {
        "view_selection": {
            "dataset": request.dataset,
            "scene": request.scene,
            "candidate_count": len(candidate_views),
            "selected_count": len(selected_views),
            "coverage_ratio": final["covered_ratio"],
        },
        "data": {"views": selected_views},
    }
    config_text = "# @package _global_\n" + yaml.safe_dump(
        config,
        sort_keys=False,
    )
    config_path = output_dir / "view_selection.yaml"
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    try:
        temporary_path.write_text(config_text, encoding="utf-8")
        os.replace(temporary_path, config_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return result
