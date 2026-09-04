from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dresg.models.diffusion.attention.features import AttentionFeatures
from dresg.models.diffusion.attention.losses import (
    align_attention_source_batch,
    query_content_loss,
    style_attention_loss,
)


def _features(
    *,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    layer_name: str = "up.attn1",
) -> AttentionFeatures:
    key = query if key is None else key
    value = key if value is None else value
    output = query if output is None else output
    return AttentionFeatures(
        layer_names=(layer_name,),
        queries=(query,),
        keys=(key,),
        values=(value,),
        outputs=(output,),
    )


def test_style_attention_loss_matches_self_attention() -> None:
    query = torch.randn((1, 1, 4, 3))
    key = torch.randn((1, 1, 4, 3))
    value = torch.randn((1, 1, 4, 3))
    self_output = F.scaled_dot_product_attention(query, key, value)
    current = _features(query=query, key=key, value=value, output=self_output)
    style = _features(query=query, key=key, value=value, output=self_output)

    loss = style_attention_loss(current, style, query_scale=1.0)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_style_attention_target_is_computed_without_autograd(monkeypatch) -> None:
    query = torch.randn((1, 1, 4, 3), requires_grad=True)
    key = torch.randn((1, 1, 4, 3), requires_grad=True)
    value = torch.randn((1, 1, 4, 3), requires_grad=True)
    current = _features(query=query, output=query.square())
    style = _features(query=query, key=key, value=value)
    original = F.scaled_dot_product_attention
    grad_modes: list[bool] = []

    def record_grad_mode(*args: torch.Tensor, **kwargs) -> torch.Tensor:
        grad_modes.append(torch.is_grad_enabled())
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record_grad_mode)
    loss = style_attention_loss(current, style, query_scale=1.0)
    loss.backward()

    assert grad_modes == [False]
    assert query.grad is not None
    assert key.grad is None
    assert value.grad is None


def test_attention_source_batch_broadcasts_one_style_reference() -> None:
    tokens = torch.randn((1, 2, 4, 3))

    aligned = align_attention_source_batch(tokens, query_batch_size=5)

    assert aligned.shape == (5, 2, 4, 3)
    assert torch.allclose(aligned[0], tokens[0])
    assert torch.allclose(aligned[-1], tokens[0])


def test_attention_source_batch_rejects_ambiguous_batch() -> None:
    with pytest.raises(ValueError, match="must be 1 or 5"):
        align_attention_source_batch(torch.randn((2, 1, 4, 3)), query_batch_size=5)


def test_content_loss_rejects_layer_identity_mismatch() -> None:
    current = _features(query=torch.zeros((1, 1, 1, 1)), layer_name="a.attn1")
    content = _features(query=torch.zeros((1, 1, 1, 1)), layer_name="b.attn1")

    with pytest.raises(ValueError, match="layers"):
        query_content_loss(current, content)


def test_attention_features_reject_mismatched_key_value_shapes() -> None:
    with pytest.raises(ValueError, match="K/V shapes"):
        _features(
            query=torch.zeros((1, 1, 2, 1)),
            key=torch.zeros((1, 1, 2, 1)),
            value=torch.zeros((1, 1, 3, 1)),
        )


def test_attention_features_reject_cross_layer_batch_or_dtype_mismatch() -> None:
    first = torch.zeros((1, 1, 2, 1))
    wrong_batch = torch.zeros((2, 1, 2, 1))
    with pytest.raises(ValueError, match="batch size"):
        AttentionFeatures(
            layer_names=("a.attn1", "b.attn1"),
            queries=(first, wrong_batch),
            keys=(first, wrong_batch),
            values=(first, wrong_batch),
            outputs=(first, wrong_batch),
        )

    wrong_dtype = torch.zeros((1, 1, 2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="one dtype"):
        AttentionFeatures(
            layer_names=("a.attn1", "b.attn1"),
            queries=(first, wrong_dtype),
            keys=(first, wrong_dtype),
            values=(first, wrong_dtype),
            outputs=(first, wrong_dtype),
        )


def test_attention_features_reject_nonfloating_tensors() -> None:
    with pytest.raises(ValueError, match="floating point"):
        _features(query=torch.zeros((1, 1, 2, 1), dtype=torch.int64))
