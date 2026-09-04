from __future__ import annotations

from collections.abc import Mapping

import torch


def _as_nchw(rgb: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if rgb.ndim == 3:
        if rgb.shape[0] != 3:
            raise ValueError(f"Expected CHW RGB tensor with C=3, got shape={tuple(rgb.shape)}")
        return rgb.unsqueeze(0), True
    if rgb.ndim == 4:
        if rgb.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB tensor with C=3, got shape={tuple(rgb.shape)}")
        return rgb, False
    raise ValueError(f"Expected CHW or NCHW RGB tensor, got shape={tuple(rgb.shape)}")


def _flatten_rgb(rgb: torch.Tensor) -> torch.Tensor:
    nchw, _ = _as_nchw(rgb)
    return nchw.permute(0, 2, 3, 1).reshape(-1, 3)


def rgb_covariance_match_colors(
    source_rgb: torch.Tensor,
    style_rgb: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match RGB mean/covariance following RGB covariance color transfer.

    RGB covariance computes a global whitening/coloring affine transform over RGB
    pixels. This implementation accepts either CHW or NCHW tensors in [0, 1].
    """

    source_nchw, squeezed = _as_nchw(source_rgb)
    source_shape = source_nchw.shape
    source_pixels = _flatten_rgb(source_nchw).float()
    style_pixels = _flatten_rgb(style_rgb).to(device=source_pixels.device).float()
    if source_pixels.numel() == 0 or style_pixels.numel() == 0:
        raise ValueError("Color transfer requires non-empty source and style pixels")

    mu_c = source_pixels.mean(0, keepdim=True)
    mu_s = style_pixels.mean(0, keepdim=True)
    centered_c = source_pixels - mu_c
    centered_s = style_pixels - mu_s
    cov_c = centered_c.transpose(1, 0).matmul(centered_c) / float(source_pixels.shape[0])
    cov_s = centered_s.transpose(1, 0).matmul(centered_s) / float(style_pixels.shape[0])

    sig_c, u_c = torch.linalg.eigh(cov_c)
    sig_s, u_s = torch.linalg.eigh(cov_s)
    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, float(eps), 1.0e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, float(eps), 1.0e8)))
    transform = u_s @ scl_s @ u_s.transpose(1, 0) @ u_c @ scl_c @ u_c.transpose(1, 0)
    bias = mu_s.view(1, 3) - mu_c.view(1, 3) @ transform.T

    transferred = source_pixels @ transform.T + bias.view(1, 3)
    transferred = transferred.clamp(0.0, 1.0).view(source_shape[0], source_shape[2], source_shape[3], 3)
    transferred = transferred.permute(0, 3, 1, 2).contiguous().to(dtype=source_rgb.dtype)

    color_tf = torch.eye(4, device=source_pixels.device, dtype=torch.float32)
    color_tf[:3, :3] = transform
    color_tf[:3, 3:4] = bias.T
    if squeezed:
        transferred = transferred[0]
    return transferred, color_tf


def rgb_covariance_match_view_colors(
    images_by_view: Mapping[int, torch.Tensor],
    style_rgb: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    if not images_by_view:
        raise ValueError("images_by_view must not be empty")
    view_ids = sorted(images_by_view)
    stacked = torch.stack([images_by_view[view_id] for view_id in view_ids], dim=0)
    transferred, color_tf = rgb_covariance_match_colors(stacked, style_rgb)
    return {view_id: transferred[index].detach() for index, view_id in enumerate(view_ids)}, color_tf
