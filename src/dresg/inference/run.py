"""Post-process a completed DReSG run from its saved Hydra configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from dresg.config import DReSGConfig, to_typed_config
from dresg.data.colmap import ColmapScene, load_colmap_scene
from dresg.inference.paths import load_video_path_for_scene
from dresg.inference.video import render_scene_video
from dresg.inference.views import export_train_view_renders
from dresg.models.gs import GaussianScene, build_gaussian_scene
from dresg.utils.results import final_gaussians_path, final_video_path

_SAVED_CONFIG_RELATIVE_PATH = Path(".hydra/config.yaml")


def load_run_config(run_dir: Path) -> DReSGConfig:
    """Load one completed run's Hydra task config against the strict schema."""
    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be a pathlib.Path")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    config_path = run_dir / _SAVED_CONFIG_RELATIVE_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Saved Hydra config does not exist: {config_path}")
    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Saved Hydra config must contain a mapping: {config_path}")
    structured = OmegaConf.merge(OmegaConf.structured(DReSGConfig), loaded)
    config = to_typed_config(structured)
    config.data.output_dir = run_dir
    return config


def _resolve_device(
    config: DReSGConfig,
    override: torch.device | None,
) -> torch.device:
    if override is not None and not isinstance(override, torch.device):
        raise TypeError("device must be a torch.device or None")
    return torch.device(config.runtime.device) if override is None else override


def _load_source(config: DReSGConfig) -> ColmapScene:
    return load_colmap_scene(
        scene_dir=config.data.scene_dir,
        factor=config.data.factor,
    )


def _load_final_scene(run_dir: Path, device: torch.device) -> GaussianScene:
    return build_gaussian_scene(
        ply_path=final_gaussians_path(run_dir),
        device=device,
        optimize_geometry=False,
        optimize_quats=False,
        max_mean_delta=0.0,
        max_scale_delta=0.0,
        max_quat_delta=0.0,
    )


def render_run_train_views(
    run_dir: Path,
    *,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    """Regenerate the complete input-view render bundle for a saved run."""
    config = load_run_config(run_dir)
    resolved_device = _resolve_device(config, device)
    source = _load_source(config)
    scene = _load_final_scene(run_dir, resolved_device)
    return export_train_view_renders(
        scene=scene,
        source=source,
        out_dir=run_dir,
        render_scale=config.rendering.render_scale,
        device=resolved_device,
    )


def render_run_video(
    run_dir: Path,
    *,
    path: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    """Regenerate a saved run's video from a saved or explicit camera path."""
    config = load_run_config(run_dir)
    video_path = config.artifacts.video.path if path is None else path
    if video_path is None:
        raise ValueError(
            "Video rendering requires --path or artifacts.video.path in the saved config"
        )
    if not isinstance(video_path, Path):
        raise TypeError("Video path must be a pathlib.Path")
    resolved_device = _resolve_device(config, device)
    source = _load_source(config)
    load_video_path_for_scene(video_path, source)
    scene = _load_final_scene(run_dir, resolved_device)
    output_path = final_video_path(run_dir)
    render_scene_video(
        scene=scene,
        source=source,
        output_path=output_path,
        video=replace(config.artifacts.video, path=video_path),
        render_scale=config.rendering.render_scale,
        device=resolved_device,
    )
    return output_path
