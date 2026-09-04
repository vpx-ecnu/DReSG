"""Validated mutable per-view latent state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class ViewLatentBank:
    """Per-view `[1, C, H, W]` latents with controlled batched updates."""

    def __init__(self, latents_by_view: Mapping[int, torch.Tensor]) -> None:
        latents = dict(latents_by_view)
        if not latents:
            raise ValueError("ViewLatentBank must contain at least one view")

        first = next(iter(latents.values()))
        if first.ndim != 4 or first.shape[0] != 1:
            raise ValueError("Each view latent must have shape [1, C, H, W]")
        if not first.is_floating_point():
            raise ValueError("View latents must use a floating-point dtype")
        if any(latent.requires_grad for latent in latents.values()):
            raise ValueError("ViewLatentBank cannot retain autograd graphs")
        for view_id, latent in latents.items():
            if latent.shape != first.shape:
                raise ValueError(
                    f"View latent shapes must match; view {view_id} has {tuple(latent.shape)} "
                    f"instead of {tuple(first.shape)}"
                )
            if latent.device != first.device:
                raise ValueError("View latents must share one device")
            if latent.dtype != first.dtype:
                raise ValueError("View latents must share one dtype")
        self._latents = latents
        self._shape = tuple(first.shape[1:])
        self._device = first.device
        self._dtype = first.dtype

    @property
    def view_ids(self) -> tuple[int, ...]:
        return tuple(self._latents)

    def __getitem__(self, view_id: int) -> torch.Tensor:
        try:
            return self._latents[view_id]
        except KeyError as error:
            raise KeyError(f"Unknown latent view ID: {view_id}") from error

    def batch(self, view_ids: Sequence[int]) -> torch.Tensor:
        return torch.cat([self[view_id] for view_id in view_ids], dim=0)

    def replace_batch(
        self,
        view_ids: Sequence[int],
        latents: torch.Tensor,
    ) -> None:
        ids = tuple(view_ids)
        if any(view_id not in self._latents for view_id in ids):
            missing = [view_id for view_id in ids if view_id not in self._latents]
            raise KeyError(f"Unknown latent view IDs: {missing}")
        expected_shape = (len(ids), *self._shape)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Replacement latent batch must have shape {expected_shape}, "
                f"got {tuple(latents.shape)}"
            )
        if latents.device != self._device or latents.dtype != self._dtype:
            raise ValueError("Replacement latents must preserve bank device and dtype")
        if latents.requires_grad:
            raise ValueError("ViewLatentBank cannot retain autograd graphs")
        for view_id, latent in zip(ids, latents.split(1, dim=0), strict=True):
            self._latents[view_id] = latent
