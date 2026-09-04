"""Execute diffusion-guidance timesteps scheduled by the trainer."""

from __future__ import annotations

import torch

from dresg.config import GuidanceOptimizationConfig
from dresg.models.diffusion import DiffusionGuidance
from dresg.training.optimization.guidance import GuidanceBatchOptimizer, GuidanceMetrics
from dresg.utils.runtime_metrics import RuntimeMetricsTracker


class GuidanceStage:
    """Own the training dependencies used across guidance timesteps."""

    def __init__(
        self,
        guidance: DiffusionGuidance,
        optimization: GuidanceOptimizationConfig,
        runtime_metrics: RuntimeMetricsTracker,
    ) -> None:
        self._guidance = guidance
        self._optimization = optimization
        self._runtime_metrics = runtime_metrics

    def run(self, timestep: torch.Tensor) -> list[GuidanceMetrics]:
        profiler = self._runtime_metrics.profile_guidance_step_start()
        timestep_state = self._guidance.prepare_timestep(timestep)
        step_metrics: list[GuidanceMetrics] = []
        view_ids = self._guidance.view_ids
        for start in range(0, len(view_ids), self._optimization.view_batch_size):
            batch = self._guidance.prepare_batch(
                timestep_state,
                view_ids[start : start + self._optimization.view_batch_size],
            )
            result = GuidanceBatchOptimizer(
                self._guidance,
                batch,
                self._optimization,
            ).run()
            step_metrics.append(result)
        self._runtime_metrics.record_guidance_step(profiler.finish())
        return step_metrics
