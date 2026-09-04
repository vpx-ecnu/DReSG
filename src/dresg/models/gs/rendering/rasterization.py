"""Canonical gsplat rasterization for validated Gaussian and camera tensors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F

RasterizationResult = tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
RasterizationFn = Callable[..., RasterizationResult]
RenderMode = Literal["RGB", "RGB+ED"]


def _default_rasterization(*args, **kwargs) -> RasterizationResult:
    from gsplat.rendering import rasterization

    return rasterization(*args, **kwargs)


def rasterize_gaussians(
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales_log: torch.Tensor,
    opacities_logit: torch.Tensor,
    colors: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int | None,
    render_mode: RenderMode,
    packed: bool,
    rasterization_fn: RasterizationFn | None = None,
) -> RasterizationResult:
    """Transform validated DReSG tensors and call gsplat."""
    rasterize = _default_rasterization if rasterization_fn is None else rasterization_fn
    return rasterize(
        means=means,
        quats=F.normalize(quats, dim=-1),
        scales=torch.exp(scales_log),
        opacities=torch.sigmoid(opacities_logit),
        colors=colors,
        viewmats=torch.linalg.inv(c2w),
        Ks=K,
        width=width,
        height=height,
        packed=packed,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        render_mode=render_mode,
        sh_degree=sh_degree,
    )
