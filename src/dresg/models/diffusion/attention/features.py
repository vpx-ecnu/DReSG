"""Typed self-attention features exchanged by diffusion guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AttentionFeatures:
    """Structure-of-arrays attention capture with explicit layer identity."""

    layer_names: tuple[str, ...]
    queries: tuple[torch.Tensor, ...]
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        count = len(self.layer_names)
        if count == 0:
            raise ValueError("Attention capture must contain at least one layer")
        if len(set(self.layer_names)) != count:
            raise ValueError("Attention layer names must be unique")
        if not (
            count
            == len(self.queries)
            == len(self.keys)
            == len(self.values)
            == len(self.outputs)
        ):
            raise ValueError("Attention layer and Q/K/V/output counts must match")

        for layer_name, query, key, value, output in zip(
            self.layer_names,
            self.queries,
            self.keys,
            self.values,
            self.outputs,
            strict=True,
        ):
            tensors = (query, key, value, output)
            if any(tensor.ndim != 4 for tensor in tensors):
                raise ValueError(
                    f"Attention features for {layer_name} must have shape [B, H, N, C]"
                )
            if key.shape != value.shape:
                raise ValueError(
                    f"Attention K/V shapes must match for {layer_name}: "
                    f"K={tuple(key.shape)} V={tuple(value.shape)}"
                )
            if query.shape != output.shape:
                raise ValueError(
                    f"Attention query/output shapes must match for {layer_name}: "
                    f"Q={tuple(query.shape)} output={tuple(output.shape)}"
                )
            if query.shape[:2] != key.shape[:2] or query.shape[3] != key.shape[3]:
                raise ValueError(
                    f"Attention Q/K head contracts do not match for {layer_name}"
                )
            first = tensors[0]
            if not first.is_floating_point():
                raise ValueError(f"Attention features must be floating point for {layer_name}")
            if any(tensor.device != first.device for tensor in tensors[1:]):
                raise ValueError(f"Attention features must share one device for {layer_name}")
            if any(tensor.dtype != first.dtype for tensor in tensors[1:]):
                raise ValueError(f"Attention features must share one dtype for {layer_name}")

        batch_size = self.queries[0].shape[0]
        device = self.queries[0].device
        dtype = self.queries[0].dtype
        if any(query.shape[0] != batch_size for query in self.queries[1:]):
            raise ValueError("All captured attention layers must share one batch size")
        if any(query.device != device for query in self.queries[1:]):
            raise ValueError("All captured attention layers must share one device")
        if any(query.dtype != dtype for query in self.queries[1:]):
            raise ValueError("All captured attention layers must share one dtype")

    def require_same_layers(self, other: AttentionFeatures, *, label: str) -> None:
        if self.layer_names != other.layer_names:
            raise ValueError(
                f"Attention layers for current and {label} features must match: "
                f"current={self.layer_names} {label}={other.layer_names}"
            )
