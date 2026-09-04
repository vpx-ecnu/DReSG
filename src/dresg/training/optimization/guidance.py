"""Training-owned optimization of one diffusion-guidance latent batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dresg.config import GuidanceOptimizationConfig
from dresg.models.diffusion import DiffusionGuidance


@dataclass(frozen=True, slots=True)
class GuidanceMetrics:
    style_loss: float
    content_loss: float
    total_loss: float
    view_count: int


class GuidanceBatchOptimizer:
    """Own the transient Adam state for one prepared latent batch."""

    def __init__(
        self,
        guidance: DiffusionGuidance,
        batch: DiffusionGuidance.OptimizationBatch,
        config: GuidanceOptimizationConfig,
    ) -> None:
        self._guidance = guidance
        self._batch = batch
        self._inner_iterations = config.inner_iterations
        self._optimizer = torch.optim.Adam(
            [batch.latents],
            lr=config.learning_rate,
        )

    def run(self) -> GuidanceMetrics:
        for _ in range(self._inner_iterations):
            self._optimizer.zero_grad(set_to_none=True)
            losses = self._guidance.batch_losses(self._batch)
            losses.total.backward()
            self._optimizer.step()

        self._guidance.commit_batch(self._batch)
        values = (
            torch.stack(
                [
                    losses.style.detach(),
                    losses.content.detach(),
                    losses.total.detach(),
                ]
            )
            .float()
            .cpu()
            .tolist()
        )
        return GuidanceMetrics(
            style_loss=values[0],
            content_loss=values[1],
            total_loss=values[2],
            view_count=len(self._batch.view_ids),
        )
