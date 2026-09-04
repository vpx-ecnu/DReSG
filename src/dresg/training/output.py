"""Training progress, metrics, and final artifact output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fsum
from typing import cast

import numpy as np
import torch

from dresg.config import ArtifactsConfig, RenderingConfig
from dresg.data.cameras import Cameras
from dresg.data.colmap import ColmapScene
from dresg.inference import export_train_view_renders, render_scene_video
from dresg.models.gs import GaussianScene
from dresg.training.optimization.guidance import GuidanceMetrics
from dresg.utils.json_io import save_json
from dresg.utils.results import final_gaussians_path, final_video_path
from dresg.utils.runtime_metrics import RuntimeMetricsTracker, RuntimeSectionProfiler


def _guidance_means(
    metrics: Sequence[GuidanceMetrics],
) -> tuple[float, float, float, int]:
    view_count = sum(metric.view_count for metric in metrics)
    return (
        fsum(metric.style_loss * metric.view_count for metric in metrics) / view_count,
        fsum(metric.content_loss * metric.view_count for metric in metrics) / view_count,
        fsum(metric.total_loss * metric.view_count for metric in metrics) / view_count,
        view_count,
    )


class TrainingProgress:
    """Own training progress, metric persistence, and final output."""

    def __init__(
        self,
        runtime_metrics: RuntimeMetricsTracker,
        output_dir: Path,
    ) -> None:
        self.runtime_metrics = runtime_metrics
        self.output_dir = output_dir
        self.stage_rows: list[dict[str, object]] = []
        self.guidance_metrics_since_prefix: list[GuidanceMetrics] = []
        self.completed_guidance_steps = 0
        self.current_prefix = 0
        self.latest_guidance_metrics: dict[str, float] = {}
        self.status = "running"
        self.aggregate: dict[str, object] = {}
        self.final_parameter_stats: dict[str, object] = {}
        (self.output_dir / "aggregate_metrics.json").unlink(missing_ok=True)
        self._write_summary()

    def record_guidance(
        self,
        *,
        prefix: int,
        metrics: Sequence[GuidanceMetrics],
    ) -> GuidanceMetrics:
        if prefix <= self.completed_guidance_steps:
            raise ValueError("Guidance prefix must exceed completed guidance steps")
        if not metrics:
            raise ValueError("Guidance metrics must not be empty")
        style_loss, content_loss, total_loss, view_count = _guidance_means(metrics)
        latest = GuidanceMetrics(
            style_loss=style_loss,
            content_loss=content_loss,
            total_loss=total_loss,
            view_count=view_count,
        )
        self.guidance_metrics_since_prefix.extend(metrics)
        self.completed_guidance_steps += 1
        self.current_prefix = prefix
        self.latest_guidance_metrics = {
            "style_loss": latest.style_loss,
            "content_loss": latest.content_loss,
            "total_loss": latest.total_loss,
        }
        self._write_summary()
        return latest

    def record_stage(self, metrics: dict[str, object]) -> dict[str, object]:
        prefix = cast(int, metrics["prefix_length"])
        if prefix != self.completed_guidance_steps:
            raise ValueError("Stage prefix must match completed guidance steps")
        style_loss, content_loss, _total_loss, _view_count = _guidance_means(
            self.guidance_metrics_since_prefix
        )
        row = {
            "stage_index": len(self.stage_rows) + 1,
            "prefix_length": prefix,
            "timestep": metrics["timestep"],
            "guidance_style_loss": style_loss,
            "guidance_content_loss": content_loss,
        }
        row.update(
            {
                key: value
                for key, value in metrics.items()
                if key not in {"stage_index", "prefix_length", "timestep"}
            }
        )
        self.stage_rows.append(row)
        self.guidance_metrics_since_prefix.clear()
        self._write_summary()
        return row

    def update_final_stage(self, metrics: Mapping[str, object]) -> None:
        if not self.stage_rows:
            raise RuntimeError("Cannot update the final stage before feedback")
        self.stage_rows[-1].update(metrics)
        self._write_summary()

    def finalize(
        self,
        *,
        scene: GaussianScene,
        source: ColmapScene,
        cameras: Cameras,
        artifacts: ArtifactsConfig,
        rendering: RenderingConfig,
    ) -> None:
        artifact_metrics = self._create_final_artifacts(
            scene=scene,
            source=source,
            cameras=cameras,
            artifacts=artifacts,
            rendering=rendering,
        )
        self.update_final_stage(artifact_metrics)
        self.aggregate = build_aggregate(self.stage_rows)
        self.final_parameter_stats = scene.parameter_stats()
        self.status = "complete"
        save_json(self.output_dir / "aggregate_metrics.json", self.aggregate)
        self._write_summary()

    def _create_final_artifacts(
        self,
        *,
        scene: GaussianScene,
        source: ColmapScene,
        cameras: Cameras,
        artifacts: ArtifactsConfig,
        rendering: RenderingConfig,
    ) -> dict[str, float]:
        device = cameras.c2w.device
        scene.save_ply(final_gaussians_path(self.output_dir))
        metrics = maybe_export_train_views(
            artifacts=artifacts,
            rendering=rendering,
            scene=scene,
            source=source,
            out_dir=self.output_dir,
            device=device,
        )
        metrics.update(
            maybe_render_final_video(
                artifacts=artifacts,
                rendering=rendering,
                scene=scene,
                source=source,
                out_dir=self.output_dir,
                device=device,
            )
        )
        return metrics

    def _write_summary(self) -> None:
        summary: dict[str, object] = {
            "method": "dresg",
            "status": self.status,
            "completed_guidance_steps": self.completed_guidance_steps,
            "current_prefix": self.current_prefix,
            "latest_guidance": self.latest_guidance_metrics,
            "rows": self.stage_rows,
        }
        if self.status == "complete":
            summary["aggregate"] = self.aggregate
            summary["final_parameter_stats"] = self.final_parameter_stats
        save_json(self.output_dir / "summary.json", summary)


def build_aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Cannot aggregate an empty training history")

    def mean_value(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    fit_rows = [cast(dict[str, float], row["fit"]) for row in rows]
    peak_allocated_mb = max(
        max(
            cast(float, row["peak_allocated_mb"]),
            fit["fit_peak_allocated_mb"],
            cast(float, row.get("post_color_transfer_peak_allocated_mb", 0.0)),
        )
        for row, fit in zip(rows, fit_rows, strict=True)
    )
    final_row = rows[-1]
    final_fit = fit_rows[-1]
    post_color_transfer_elapsed = cast(
        float,
        final_row.get("post_color_transfer_elapsed_sec", 0.0),
    )
    return {
        "num_stages": len(rows),
        "mean_teacher_l1": mean_value("teacher_l1"),
        "final_teacher_l1": final_row["teacher_l1"],
        "mean_projection_gap_l1": mean_value("projection_gap_l1"),
        "final_projection_gap_l1": final_row["projection_gap_l1"],
        "mean_fit_l1": float(np.mean([fit["final_l1"] for fit in fit_rows])),
        "final_fit_l1": final_fit["final_l1"],
        "final_fit_total": final_fit["final_total"],
        "mean_content3d_loss": float(
            np.mean([fit["final_content3d_loss"] for fit in fit_rows])
        ),
        "final_content3d_loss": final_fit["final_content3d_loss"],
        "train_measured_elapsed_sec": (
            cast(float, final_row["train_measured_elapsed_sec"])
            + post_color_transfer_elapsed
        ),
        "post_color_transfer_elapsed_sec": post_color_transfer_elapsed,
        "video_elapsed_sec": final_row["video_elapsed_sec"],
        "peak_allocated_mb": peak_allocated_mb,
    }


def _profile_optional_artifact(
    *,
    enabled: bool,
    device: torch.device,
    metric_prefix: str,
    action: Callable[[], None],
) -> dict[str, float]:
    if not enabled:
        return {
            f"{metric_prefix}_elapsed_sec": 0.0,
            f"{metric_prefix}_peak_allocated_mb": 0.0,
        }

    profiler = RuntimeSectionProfiler(device)
    profiler.start()
    action()
    runtime = profiler.finish()
    return {
        f"{metric_prefix}_elapsed_sec": runtime.elapsed_sec,
        f"{metric_prefix}_peak_allocated_mb": runtime.peak_allocated_mb,
    }


def maybe_render_final_video(
    *,
    artifacts: ArtifactsConfig,
    rendering: RenderingConfig,
    scene: GaussianScene,
    source: ColmapScene,
    out_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    return _profile_optional_artifact(
        enabled=artifacts.video.path is not None,
        device=device,
        metric_prefix="video",
        action=lambda: render_scene_video(
            scene=scene,
            source=source,
            output_path=final_video_path(out_dir),
            video=artifacts.video,
            render_scale=rendering.render_scale,
            device=device,
        ),
    )


def maybe_export_train_views(
    *,
    artifacts: ArtifactsConfig,
    rendering: RenderingConfig,
    scene: GaussianScene,
    source: ColmapScene,
    out_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    if not artifacts.save_train_views:
        return {}
    metrics = export_train_view_renders(
        scene=scene,
        source=source,
        out_dir=out_dir,
        render_scale=rendering.render_scale,
        device=device,
    )
    return {
        "infer_elapsed_sec": float(metrics["pure_infer_elapsed_sec"]),
        "infer_fps": float(metrics["infer_fps"]),
        "infer_peak_mem_mb": float(metrics["infer_peak_mem_mb"]),
    }


def build_training_progress(
    device: torch.device,
    output_dir: Path,
) -> TrainingProgress:
    """Build the persistent output owner for one run."""
    return TrainingProgress(RuntimeMetricsTracker(device), output_dir)
