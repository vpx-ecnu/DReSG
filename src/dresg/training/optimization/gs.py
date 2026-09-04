"""Persistent multi-view Gaussian optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from dresg.config import AppearanceOptimizationConfig, ImageLossConfig
from dresg.data.cameras import Cameras
from dresg.models.gs import GaussianScene
from dresg.models.gs.fitting import (
    DinoPatchContentLoss,
    compute_appearance_losses,
    fuse_appearance_gradients,
)
from dresg.utils.runtime_metrics import RuntimeSectionProfiler


@dataclass(frozen=True, slots=True)
class FitMetrics:
    final_total: float
    final_l1: float
    final_content3d_loss: float
    elapsed_sec: float
    fit_peak_allocated_mb: float

    def to_dict(self) -> dict[str, float]:
        return {
            "final_total": self.final_total,
            "final_l1": self.final_l1,
            "final_content3d_loss": self.final_content3d_loss,
            "fit_elapsed_sec": self.elapsed_sec,
            "fit_peak_allocated_mb": self.fit_peak_allocated_mb,
        }


class GaussianOptimizer:
    """Own the persistent Adam state for one Gaussian scene."""

    def __init__(
        self,
        scene: GaussianScene,
        cameras: Cameras,
        config: AppearanceOptimizationConfig,
    ) -> None:
        self._scene = scene
        self._cameras = cameras
        self._appearance = scene.appearance_parameters()[0]
        self._geometry = tuple(scene.geometry_parameters())
        groups: list[dict[str, object]] = [
            {"params": [self._appearance], "lr": config.lr}
        ]
        mean_scale = scene.geometry_mean_scale_parameters()
        if mean_scale:
            groups.append({"params": mean_scale, "lr": config.geometry_lr})
        quaternions = scene.geometry_quat_parameters()
        if quaternions:
            groups.append(
                {
                    "params": quaternions,
                    "lr": (
                        config.geometry_lr
                        if config.geometry_quat_lr is None
                        else config.geometry_quat_lr
                    ),
                }
            )
        self._optimizer = torch.optim.Adam(groups)

    def run(
        self,
        *,
        teachers_by_view: Mapping[int, torch.Tensor],
        base_renders_by_view: Mapping[int, torch.Tensor] | None,
        image_loss: ImageLossConfig,
        content_loss: DinoPatchContentLoss | None,
        fit_steps: int,
        appearance_update_rule: str,
        update_geometry: bool,
    ) -> FitMetrics:
        geometry_requires_grad = [
            parameter.requires_grad for parameter in self._geometry
        ]
        if not update_geometry:
            for parameter in self._geometry:
                parameter.requires_grad_(False)

        final_l1 = 0.0
        final_content3d_loss = 0.0
        final_total = 0.0
        profiler = RuntimeSectionProfiler(self._appearance.device)
        profiler.start()
        for step in range(1, fit_steps + 1):
            metric_rows = self._step(
                teachers_by_view=teachers_by_view,
                base_renders_by_view=base_renders_by_view,
                image_loss=image_loss,
                content_loss=content_loss,
                appearance_update_rule=appearance_update_rule,
                update_geometry=update_geometry,
            )
            if step == fit_steps:
                final_l1, final_content3d_loss, final_total = (
                    torch.stack(metric_rows).mean(dim=0).cpu().tolist()
                )

        if not update_geometry:
            self._optimizer.zero_grad(set_to_none=True)
            for parameter, requires_grad in zip(
                self._geometry,
                geometry_requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(requires_grad)
        runtime = profiler.finish()

        return FitMetrics(
            final_total=final_total,
            final_l1=final_l1,
            final_content3d_loss=final_content3d_loss,
            elapsed_sec=runtime.elapsed_sec,
            fit_peak_allocated_mb=runtime.peak_allocated_mb,
        )

    def _step(
        self,
        *,
        teachers_by_view: Mapping[int, torch.Tensor],
        base_renders_by_view: Mapping[int, torch.Tensor] | None,
        image_loss: ImageLossConfig,
        content_loss: DinoPatchContentLoss | None,
        appearance_update_rule: str,
        update_geometry: bool,
    ) -> list[torch.Tensor]:
        appearance = self._appearance
        geometry = self._geometry if update_geometry else ()
        view_count = len(self._cameras)
        project_appearance = appearance_update_rule == "pcgrad"
        appearance_gradients = (
            []
            if project_appearance
            else [torch.zeros_like(appearance.detach(), dtype=torch.float32)]
        )
        geometry_gradients = [
            torch.zeros_like(parameter.detach(), dtype=torch.float32)
            for parameter in geometry
        ]
        metric_rows: list[torch.Tensor] = []
        self._optimizer.zero_grad(set_to_none=True)

        for camera in self._cameras:
            view_id = camera.view_index
            losses = compute_appearance_losses(
                render_rgb=self._scene.render(camera, clamp=False),
                teacher_rgb=teachers_by_view[view_id],
                base_render_rgb=(
                    None
                    if base_renders_by_view is None
                    else base_renders_by_view[view_id]
                ),
                content_loss=content_loss,
                view_id=view_id,
                config=image_loss,
            )
            view_gradients = torch.autograd.grad(
                losses.total / float(view_count),
                (appearance, *geometry),
            )
            appearance_gradient, *geometry_view_gradients = view_gradients
            if project_appearance:
                appearance_gradients.append(appearance_gradient.detach().float())
            else:
                appearance_gradients[0].add_(appearance_gradient.detach().float())
            for gradient_sum, view_gradient in zip(
                geometry_gradients,
                geometry_view_gradients,
                strict=True,
            ):
                gradient_sum.add_(view_gradient.detach().float())
            metric_rows.append(
                torch.stack(
                    [
                        losses.l1.detach(),
                        losses.content3d.detach(),
                        losses.total.detach(),
                    ]
                ).float()
            )

        appearance_gradient = fuse_appearance_gradients(
            torch.stack(appearance_gradients),
            rule=appearance_update_rule,
        )
        if not project_appearance:
            appearance_gradient = appearance_gradient / float(view_count)
        appearance.grad = appearance_gradient.to(appearance)
        for parameter, gradient_sum in zip(
            geometry,
            geometry_gradients,
            strict=True,
        ):
            parameter.grad = (gradient_sum / float(view_count)).to(parameter)

        self._optimizer.step()
        self._scene.apply_parameter_constraints()
        return metric_rows
