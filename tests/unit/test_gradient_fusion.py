from __future__ import annotations

import pytest
import torch

from dresg.models.gs.fitting.fusion import (
    fuse_appearance_gradients,
)


def _fuse(
    grad_stack: torch.Tensor,
    *,
    rule: str,
) -> torch.Tensor:
    return fuse_appearance_gradients(
        grad_stack,
        rule=rule,
    )


def test_standard_fusion_is_arithmetic_mean() -> None:
    grad_stack = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    fused = _fuse(grad_stack, rule="standard")

    assert torch.allclose(fused, grad_stack.mean(dim=0))


def test_pcgrad_removes_two_view_negative_conflict() -> None:
    grad_stack = torch.tensor(
        [
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )

    fused = _fuse(grad_stack, rule="pcgrad")

    assert torch.allclose(fused, torch.zeros_like(fused), atol=1e-6)


def test_pcgrad_matches_sequential_projection_reference() -> None:
    grad_stack = torch.tensor(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[-0.5, 1.0], [2.0, -1.0]],
            [[0.25, 0.5], [-3.0, 1.5]],
        ]
    )
    references = grad_stack.reshape(3, -1).float()
    projected = references.clone()
    eps = max(float(references.norm(dim=1).mean()) * 1.0e-8, 1.0e-12)
    for index in range(3):
        gradient = projected[index]
        for reference_index in range(3):
            if index == reference_index:
                continue
            reference = references[reference_index]
            dot = torch.dot(gradient, reference)
            if float(dot) < 0.0:
                gradient = gradient - dot / torch.dot(reference, reference).clamp_min(eps) * reference
        projected[index] = gradient
    expected = projected.mean(dim=0).reshape_as(grad_stack[0])

    fused = _fuse(grad_stack, rule="pcgrad")

    torch.testing.assert_close(fused, expected)


def test_single_view_gradient_is_preserved() -> None:
    grad_stack = torch.tensor([[[1.0, -2.0]]])

    fused = _fuse(grad_stack, rule="pcgrad")

    torch.testing.assert_close(fused, grad_stack[0])



def test_invalid_appearance_update_rule_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported appearance_update.rule"):
        _fuse(torch.ones((2, 1, 2)), rule="not_a_rule")
