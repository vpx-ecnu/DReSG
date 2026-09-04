from __future__ import annotations

from types import SimpleNamespace

import torch

from dresg.config import GuidanceOptimizationConfig
from dresg.training.optimization.guidance import GuidanceBatchOptimizer


class _Guidance:
    def __init__(self) -> None:
        self.committed = False

    def batch_losses(self, batch):
        style = (batch.latents - 1.0).square().mean()
        content = batch.latents.square().mean()
        return SimpleNamespace(
            style=style,
            content=content,
            total=style + 0.1 * content,
        )

    def commit_batch(self, _batch) -> None:
        self.committed = True


def test_guidance_fit_owns_adam_backward_and_commit() -> None:
    guidance = _Guidance()
    batch = SimpleNamespace(
        latents=torch.zeros(2, 1, requires_grad=True),
        view_ids=(4, 7),
    )
    optimization = GuidanceOptimizationConfig(
        num_inference_steps=1,
        learning_rate=0.1,
        inner_iterations=1,
        view_batch_size=2,
    )

    metrics = GuidanceBatchOptimizer(guidance, batch, optimization).run()

    torch.testing.assert_close(batch.latents, torch.full_like(batch.latents, 0.1))
    assert guidance.committed
    assert metrics.style_loss == 1.0
    assert metrics.content_loss == 0.0
    assert metrics.total_loss == 1.0
    assert metrics.view_count == 2
