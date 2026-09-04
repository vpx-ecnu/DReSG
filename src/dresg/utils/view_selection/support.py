"""Visible-Gaussian support estimation for active-view selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dresg.data.cameras import CameraView
from dresg.models.gs import GaussianScene
from dresg.models.gs.rendering.rasterization import rasterize_gaussians


@dataclass(frozen=True)
class SparseViewSupport:
    view_index: int
    gaussian_indices: torch.Tensor
    values: torch.Tensor
    visible_samples: int
    depth_rejected: int

    def __post_init__(self) -> None:
        if self.gaussian_indices.ndim != 1 or self.values.ndim != 1:
            raise ValueError("Sparse support indices and values must be one-dimensional")
        if self.gaussian_indices.shape != self.values.shape:
            raise ValueError("Sparse support indices and values must have matching shapes")


def sample_image_features(
    image_bchw: torch.Tensor,
    coords_xy: torch.Tensor,
    *,
    width: int,
    height: int,
    chunk_size: int = 32768,
) -> torch.Tensor:
    if coords_xy.numel() == 0:
        return image_bchw.new_zeros((0, image_bchw.shape[1]))
    grid_dtype = image_bchw.dtype
    if image_bchw.device.type == "cpu" and grid_dtype not in {
        torch.float32,
        torch.float64,
    }:
        grid_dtype = torch.float32
    outputs = []
    for start in range(0, coords_xy.shape[0], chunk_size):
        coords = coords_xy[start : start + chunk_size]
        x = coords[:, 0] / max(width - 1, 1) * 2.0 - 1.0
        y = coords[:, 1] / max(height - 1, 1) * 2.0 - 1.0
        grid = (
            torch.stack([x, y], dim=-1)
            .to(dtype=grid_dtype)
            .view(
                1,
                -1,
                1,
                2,
            )
        )
        sampled = F.grid_sample(
            image_bchw,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        outputs.append(sampled[0, :, :, 0].transpose(0, 1).contiguous())
    return torch.cat(outputs, dim=0)


def sample_footprint_features(
    image_bchw: torch.Tensor,
    coords_xy: torch.Tensor,
    radii_xy: torch.Tensor,
    *,
    width: int,
    height: int,
    grid_size: int,
    radius_scale: float,
) -> torch.Tensor:
    if coords_xy.numel() == 0:
        return image_bchw.new_zeros((0, image_bchw.shape[1]))
    steps = torch.linspace(
        -1.0,
        1.0,
        steps=grid_size,
        device=coords_xy.device,
        dtype=coords_xy.dtype,
    )
    offset_y, offset_x = torch.meshgrid(steps, steps, indexing="ij")
    offsets = torch.stack(
        [offset_x.reshape(-1), offset_y.reshape(-1)],
        dim=-1,
    )
    weights = torch.exp(-offsets.square().sum(dim=-1))
    weights = weights / weights.sum().clamp_min(1e-6)
    weight_values = weights.cpu().tolist()
    output = image_bchw.new_zeros((coords_xy.shape[0], image_bchw.shape[1]))
    for weight, offset in zip(weight_values, offsets, strict=True):
        coords = coords_xy + offset.unsqueeze(0) * radii_xy * radius_scale
        output.add_(
            sample_image_features(
                image_bchw,
                coords,
                width=width,
                height=height,
            ),
            alpha=weight,
        )
    return output


@torch.no_grad()
def compute_view_support(
    *,
    scene: GaussianScene,
    camera: CameraView,
    pool_grid_size: int,
    pool_radius_scale: float,
    depth_gate: bool,
    depth_tolerance: float,
    depth_tolerance_ratio: float,
    rasterization_fn: Callable | None = None,
) -> SparseViewSupport:
    renders, alphas, info = rasterize_gaussians(
        means=scene.means(),
        quats=scene.quats(),
        scales_log=scene.scales_log(),
        opacities_logit=scene.opacities_logit(),
        colors=scene.colors(),
        c2w=camera.c2w.unsqueeze(0),
        K=camera.K.unsqueeze(0),
        width=camera.width,
        height=camera.height,
        packed=True,
        render_mode="RGB+ED",
        sh_degree=None,
        rasterization_fn=rasterization_fn,
    )
    view_index = camera.view_index
    total_gaussians = scene.means().shape[0]
    gaussian_ids = info["gaussian_ids"].long()
    if gaussian_ids.numel() == 0:
        return SparseViewSupport(
            view_index=view_index,
            gaussian_indices=torch.empty(0, dtype=torch.long),
            values=torch.empty(0, dtype=torch.float32),
            visible_samples=0,
            depth_rejected=0,
        )

    alpha = alphas[0, ..., 0]
    depth = renders[0, ..., 3]
    means2d = info["means2d"]
    radii = info["radii"].float()
    if radii.ndim == 1:
        radii = radii[:, None].expand(-1, 2)
    opacities = info["opacities"].reshape(-1).to(dtype=means2d.dtype)

    support = sample_footprint_features(
        alpha[None, None],
        means2d,
        radii,
        width=camera.width,
        height=camera.height,
        grid_size=pool_grid_size,
        radius_scale=pool_radius_scale,
    )[:, 0]
    support = (support * opacities.clamp_min(1e-6)).float()

    depth_rejected = 0
    if depth_gate:
        depth_samples = sample_footprint_features(
            depth[None, None],
            means2d,
            radii,
            width=camera.width,
            height=camera.height,
            grid_size=pool_grid_size,
            radius_scale=pool_radius_scale,
        )[:, 0].float()
        gaussian_depths = info["depths"].reshape(-1).float()
        finite_depth = torch.isfinite(depth_samples) & (depth_samples > 0)
        if finite_depth.any():
            depth_scale = torch.median(depth_samples[finite_depth]).clamp_min(1e-6)
            tolerance = torch.maximum(
                depth_samples.new_tensor(depth_tolerance),
                depth_tolerance_ratio * depth_scale,
            )
        else:
            tolerance = depth_tolerance
        front_visible = finite_depth & (gaussian_depths <= depth_samples + tolerance)
        depth_rejected = (~front_visible).sum().item()
        support = support * front_visible.to(dtype=support.dtype)

    dense_support = torch.zeros(
        total_gaussians,
        device=support.device,
        dtype=torch.float32,
    )
    dense_support.index_add_(0, gaussian_ids, support)
    nonzero = dense_support > 0
    return SparseViewSupport(
        view_index=view_index,
        gaussian_indices=nonzero.nonzero(as_tuple=False).squeeze(1).cpu(),
        values=dense_support[nonzero].cpu(),
        visible_samples=gaussian_ids.numel(),
        depth_rejected=depth_rejected,
    )
