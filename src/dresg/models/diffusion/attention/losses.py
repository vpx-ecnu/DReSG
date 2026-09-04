"""Strict attention losses used to optimize teacher latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dresg.models.diffusion.attention.features import AttentionFeatures


@dataclass(frozen=True, slots=True)
class GuidanceLosses:
    style: torch.Tensor
    content: torch.Tensor
    total: torch.Tensor


def align_attention_source_batch(
    tokens: torch.Tensor,
    *,
    query_batch_size: int,
) -> torch.Tensor:
    """Broadcast a single style reference or preserve per-view content features."""
    batch_size = tokens.shape[0]
    if batch_size == query_batch_size:
        return tokens
    if batch_size == 1:
        return tokens.expand(query_batch_size, -1, -1, -1)
    raise ValueError(f"Attention source batch must be 1 or {query_batch_size}, got {batch_size}")


def query_content_loss(
    current: AttentionFeatures,
    content: AttentionFeatures,
) -> torch.Tensor:
    current.require_same_layers(content, label="content")
    loss = current.queries[0].new_zeros(())
    for query, content_query in zip(current.queries, content.queries, strict=True):
        aligned_content = align_attention_source_batch(
            content_query,
            query_batch_size=query.shape[0],
        )
        if query.shape != aligned_content.shape:
            raise ValueError(
                "Current/content query shapes must match after batch alignment: "
                f"current={tuple(query.shape)} content={tuple(aligned_content.shape)}"
            )
        loss = loss + F.l1_loss(query, aligned_content.detach())
    return loss


def style_attention_loss(
    current: AttentionFeatures,
    style: AttentionFeatures,
    *,
    query_scale: float,
) -> torch.Tensor:
    current.require_same_layers(style, label="style")
    loss = current.queries[0].new_zeros(())
    for layer_name, query, style_key, style_value, self_output in zip(
        current.layer_names,
        current.queries,
        style.keys,
        style.values,
        current.outputs,
        strict=True,
    ):
        key = align_attention_source_batch(
            style_key,
            query_batch_size=query.shape[0],
        )
        value = align_attention_source_batch(
            style_value,
            query_batch_size=query.shape[0],
        )
        if key.shape != value.shape:
            raise ValueError(
                f"Style K/V shapes must match for {layer_name}: K={tuple(key.shape)} V={tuple(value.shape)}"
            )
        if query.shape[:2] != key.shape[:2] or query.shape[3] != key.shape[3]:
            raise ValueError(f"Current/style attention head contracts do not match for {layer_name}")
        with torch.no_grad():
            target_output = F.scaled_dot_product_attention(
                query * query_scale,
                key,
                value,
            )
        loss = loss + F.l1_loss(self_output, target_output)
    return loss


def attention_guidance_losses(
    *,
    current: AttentionFeatures,
    style: AttentionFeatures,
    content: AttentionFeatures,
    content_weight: float,
    query_scale: float,
) -> GuidanceLosses:
    style_loss = style_attention_loss(
        current,
        style,
        query_scale=query_scale,
    )
    content_loss = query_content_loss(current, content)
    total_loss = style_loss + content_weight * content_loss
    return GuidanceLosses(
        style=style_loss,
        content=content_loss,
        total=total_loss,
    )
