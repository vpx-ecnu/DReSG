"""Execute the scheduled post-color-transfer fitting stage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from dresg.config import ImageLossConfig
from dresg.data.cameras import Cameras
from dresg.models.gs import GaussianScene
from dresg.training.optimization.gs import GaussianOptimizer
from dresg.utils.color_transfer import rgb_covariance_match_view_colors
from dresg.utils.images import load_rgb_image
from dresg.utils.runtime_metrics import RuntimeSectionProfiler, memory_snapshot


class ColorStage:
    """Own the scene runtime used by the final color-transfer stage."""

    def __init__(
        self,
        scene: GaussianScene,
        cameras: Cameras,
        optimizer: GaussianOptimizer,
        *,
        style_image: Path,
        image_loss: ImageLossConfig,
    ) -> None:
        self._scene = scene
        self._cameras = cameras
        self._optimizer = optimizer
        self._style_image = style_image
        self._image_loss = image_loss

    def run(self, *, fit_steps: int) -> dict[str, float | int]:
        device = self._cameras.c2w.device
        profiler = RuntimeSectionProfiler(device)
        profiler.start()
        pre_transfer_renders = self._scene.render_current_images(self._cameras)
        style_rgb = load_rgb_image(self._style_image, device=device)
        targets_by_view, _ = rgb_covariance_match_view_colors(
            pre_transfer_renders,
            style_rgb,
        )

        pre_fit_peak = memory_snapshot(device).peak_allocated_mb
        fit_metrics = self._optimizer.run(
            teachers_by_view=targets_by_view,
            base_renders_by_view=None,
            image_loss=self._image_loss,
            content_loss=None,
            fit_steps=fit_steps,
            appearance_update_rule="standard",
            update_geometry=False,
        )
        post_fit_renders = self._scene.render_current_images(self._cameras)
        target_l1s = self._measure_renders(
            post_fit_renders,
            targets_by_view,
        )
        runtime = profiler.finish()
        metrics: dict[str, float | int] = {
            "post_color_transfer_enabled": 1,
            "post_color_transfer_render_target_l1": (sum(target_l1s) / len(target_l1s)),
            "post_color_transfer_fit_l1": fit_metrics.final_l1,
            "post_color_transfer_fit_elapsed_sec": fit_metrics.elapsed_sec,
            "post_color_transfer_elapsed_sec": runtime.elapsed_sec,
            "post_color_transfer_peak_allocated_mb": max(
                runtime.peak_allocated_mb,
                pre_fit_peak,
                fit_metrics.fit_peak_allocated_mb,
            ),
        }
        return metrics

    def _measure_renders(
        self,
        renders_by_view: Mapping[int, torch.Tensor],
        targets_by_view: Mapping[int, torch.Tensor],
    ) -> list[float]:
        losses: list[torch.Tensor] = []
        for view_id in sorted(targets_by_view):
            losses.append(F.l1_loss(renders_by_view[view_id], targets_by_view[view_id]))
        return torch.stack(losses).float().cpu().tolist()
