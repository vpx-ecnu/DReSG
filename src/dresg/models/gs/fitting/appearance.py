"""Per-view loss composition for Gaussian appearance fitting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dresg.config import ImageLossConfig
from dresg.models.gs.fitting.dino import DinoPatchContentLoss
from dresg.models.gs.fitting.image import structural_similarity, total_variation_loss


@dataclass(frozen=True, slots=True)
class AppearanceLosses:
    """Differentiable loss components for one rendered training view."""

    total: torch.Tensor
    l1: torch.Tensor
    content3d: torch.Tensor


def _validate_rgb_chw(image: torch.Tensor, *, name: str) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"{name} must have shape [3, H, W]")
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"{name} spatial dimensions must be non-empty")
    if not image.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def compute_appearance_losses(
    *,
    render_rgb: torch.Tensor,
    teacher_rgb: torch.Tensor,
    base_render_rgb: torch.Tensor | None,
    content_loss: DinoPatchContentLoss | None,
    view_id: int,
    config: ImageLossConfig,
) -> AppearanceLosses:
    """Compose differentiable image and content losses for one rendered view."""
    _validate_rgb_chw(render_rgb, name="render_rgb")
    _validate_rgb_chw(teacher_rgb, name="teacher_rgb")
    if teacher_rgb.shape != render_rgb.shape:
        raise ValueError("teacher_rgb shape must match render_rgb")
    if teacher_rgb.device != render_rgb.device or teacher_rgb.dtype != render_rgb.dtype:
        raise ValueError("teacher_rgb must match render_rgb device and dtype")

    render_clamped = render_rgb.clamp(0.0, 1.0)
    render_bchw = render_clamped.unsqueeze(0)
    render_l1_bchw = render_rgb.unsqueeze(0) if config.l1_use_unclamped_render else render_bchw
    teacher_bchw = teacher_rgb.unsqueeze(0)
    content3d = render_clamped.new_zeros(())
    content_enabled = config.lambda_content3d > 0.0
    if content_enabled and (base_render_rgb is None) != (content_loss is None):
        raise ValueError("Enabled content loss requires both base_render_rgb and content_loss, or neither")
    if content_enabled and base_render_rgb is not None:
        _validate_rgb_chw(base_render_rgb, name="base_render_rgb")
        if base_render_rgb.device != render_clamped.device or base_render_rgb.dtype != render_clamped.dtype:
            raise ValueError("base_render_rgb must match render_rgb device and dtype")
        if base_render_rgb.shape[-2:] != render_clamped.shape[-2:]:
            base_render_rgb = F.interpolate(
                base_render_rgb.unsqueeze(0),
                size=render_clamped.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        assert content_loss is not None
        content3d = content_loss.loss(
            render_bchw,
            base_render_rgb.unsqueeze(0),
            view_id=view_id,
        )

    l1 = F.l1_loss(render_l1_bchw, teacher_bchw)
    dssim = 1.0 - structural_similarity(render_bchw, teacher_bchw)
    image_tv = total_variation_loss(render_bchw)
    total = (
        config.lambda_l1 * l1
        + config.lambda_dssim * dssim
        + config.lambda_img_tv * image_tv
        + config.lambda_content3d * content3d
    )
    return AppearanceLosses(
        total=total,
        l1=l1,
        content3d=content3d,
    )
