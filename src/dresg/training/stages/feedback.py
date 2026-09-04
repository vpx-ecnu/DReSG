"""Execute scheduled teacher-to-Gaussian residual-feedback stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean

import torch
import torch.nn.functional as F

from dresg.config import ImageLossConfig
from dresg.data.cameras import Cameras
from dresg.data.images import ViewImages
from dresg.models.diffusion import DiffusionGuidance
from dresg.models.gs import GaussianScene
from dresg.models.gs.fitting import DinoPatchContentLoss
from dresg.training.optimization.gs import GaussianOptimizer
from dresg.utils.runtime_metrics import RuntimeMetricsTracker
from dresg.utils.tensor_stats import (
    average_metric_dicts,
    image_color_stats,
    tensor_range_stats,
)


@dataclass(frozen=True, slots=True)
class _TeacherTargets:
    teachers_by_view: dict[int, torch.Tensor]
    sources_by_view: dict[int, torch.Tensor]
    residuals: list[float]


def _stage_color_diagnostics(
    *,
    scene: GaussianScene,
    cameras: Cameras,
    sources_by_view: Mapping[int, torch.Tensor],
    teachers_by_view: Mapping[int, torch.Tensor],
) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    source_stats = [
        image_color_stats("source", image) for image in sources_by_view.values()
    ]
    teacher_stats = [
        image_color_stats("teacher", image) for image in teachers_by_view.values()
    ]
    render_stats = []
    raw_render_ranges = []
    for camera in cameras:
        render_raw = scene.render(camera, clamp=False)
        render_stats.append(image_color_stats("render", render_raw.clamp(0.0, 1.0)))
        raw_render_ranges.append(tensor_range_stats("raw_render", render_raw))
    diagnostics.update(average_metric_dicts(source_stats))
    diagnostics.update(average_metric_dicts(teacher_stats))
    diagnostics.update(average_metric_dicts(render_stats))
    diagnostics.update(average_metric_dicts(raw_render_ranges))
    diagnostics.update(tensor_range_stats("appearance_rgb", scene.colors()))
    return diagnostics


class FeedbackStage:
    """Own dependencies for periodic residual-feedback execution."""

    def __init__(
        self,
        scene: GaussianScene,
        cameras: Cameras,
        optimizer: GaussianOptimizer,
        base_renders: ViewImages,
        guidance: DiffusionGuidance,
        runtime_metrics: RuntimeMetricsTracker,
        content_loss: DinoPatchContentLoss | None,
        *,
        image_loss: ImageLossConfig,
        appearance_update_rule: str,
        collect_diagnostics: bool,
    ) -> None:
        self._scene = scene
        self._cameras = cameras
        self._optimizer = optimizer
        self._base_renders = base_renders
        self._guidance = guidance
        self._runtime_metrics = runtime_metrics
        self._content_loss = content_loss
        self._image_loss = image_loss
        self._appearance_update_rule = appearance_update_rule
        self._collect_diagnostics = collect_diagnostics

    def run(
        self,
        *,
        prefix: int,
        fit_steps: int,
    ) -> dict[str, object]:
        stage_timestep = self._guidance.timesteps[prefix - 1].item()
        self._runtime_metrics.start_stage()
        targets = self._build_teacher_targets(
            teacher_scale=self._guidance.scale_at(prefix),
        )
        self._runtime_metrics.capture_stage_peak()
        fit_metrics = self._optimizer.run(
            teachers_by_view=targets.teachers_by_view,
            base_renders_by_view=self._base_renders,
            image_loss=self._image_loss,
            content_loss=self._content_loss,
            fit_steps=fit_steps,
            appearance_update_rule=self._appearance_update_rule,
            update_geometry=True,
        )
        projection_gaps = self._project_renders()
        diagnostics = (
            _stage_color_diagnostics(
                scene=self._scene,
                cameras=self._cameras,
                sources_by_view=targets.sources_by_view,
                teachers_by_view=targets.teachers_by_view,
            )
            if self._collect_diagnostics
            else None
        )
        row: dict[str, object] = {
            "prefix_length": prefix,
            "timestep": stage_timestep,
            "teacher_l1": fmean(targets.residuals),
            "projection_gap_l1": fmean(projection_gaps),
            "fit": fit_metrics.to_dict(),
            "effective_fit_steps": fit_steps,
        }
        if diagnostics is not None:
            row.update(diagnostics)
            row["parameter_stats"] = self._scene.parameter_stats()
        row.update(
            self._runtime_metrics.finish_stage(
                fit_peak_allocated_mb=fit_metrics.fit_peak_allocated_mb,
            )
        )
        row["post_color_transfer_enabled"] = 0
        return row

    def _build_teacher_targets(
        self,
        *,
        teacher_scale: float,
    ) -> _TeacherTargets:
        teachers: dict[int, torch.Tensor] = {}
        sources: dict[int, torch.Tensor] = {}
        residuals: list[torch.Tensor] = []
        with torch.no_grad():
            for camera in self._cameras:
                view_id = camera.view_index
                source = self._scene.render(camera)
                teacher = self._guidance.teacher_image(
                    view_id=view_id,
                    source_rgb=source,
                    scale=teacher_scale,
                )
                teachers[view_id] = teacher
                residuals.append(F.l1_loss(teacher, source))
                if self._collect_diagnostics:
                    sources[view_id] = source.detach()
        return _TeacherTargets(
            teachers_by_view=teachers,
            sources_by_view=sources,
            residuals=torch.stack(residuals).float().cpu().tolist(),
        )

    def _project_renders(self) -> list[float]:
        projection_gaps: list[torch.Tensor] = []
        with torch.no_grad():
            for camera in self._cameras:
                projection_gap = self._guidance.project_render(
                    view_id=camera.view_index,
                    render_rgb=self._scene.render(camera),
                )
                projection_gaps.append(projection_gap)
        return torch.stack(projection_gaps).float().cpu().tolist()
