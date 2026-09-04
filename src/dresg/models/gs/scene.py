"""Gaussian scene facade with direct-RGB appearance optimization."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.exporter import sh2rgb

from dresg.data.cameras import Cameras, CameraView
from dresg.data.images import ViewImages
from dresg.models.gs.rendering.rasterization import rasterize_gaussians
from dresg.models.gs.serialization.ply import (
    SplatTensors,
    load_gaussian_ply,
    save_gaussian_ply,
)
from dresg.models.gs.serialization.sh import rgb_to_sh0
from dresg.utils.tensor_stats import tensor_range_stats


class GaussianScene(nn.Module):
    """Scene that loads Gaussian PLY data and optimizes direct RGB."""

    def __init__(
        self,
        *,
        splats: SplatTensors,
        device: torch.device,
        optimize_geometry: bool,
        optimize_quats: bool,
        max_mean_delta: float,
        max_scale_delta: float,
        max_quat_delta: float,
    ) -> None:
        super().__init__()

        means = splats["means"]
        quats = splats["quats"]
        scales = splats["scales"]
        opacities = splats["opacities"]
        sh0 = splats["sh0"]
        self.max_mean_delta = max_mean_delta
        self.max_scale_delta = max_scale_delta
        self.max_quat_delta = max_quat_delta
        self.register_buffer(
            "base_means",
            means.to(device=device, dtype=torch.float32).contiguous(),
        )
        self.register_buffer(
            "base_quats",
            quats.to(device=device, dtype=torch.float32).contiguous(),
        )
        self.register_buffer(
            "base_scales_log",
            scales.to(device=device, dtype=torch.float32).contiguous(),
        )
        self.register_buffer(
            "base_opacities_logit",
            opacities.to(device=device, dtype=torch.float32).contiguous(),
        )
        appearance_rgb = sh2rgb(sh0.to(device=device, dtype=torch.float32)).squeeze(1)
        self.appearance_rgb = nn.Parameter(appearance_rgb.clamp(0.0, 1.0).contiguous())

        self.optimize_geometry = optimize_geometry
        self.optimize_quats = optimize_quats
        if self.optimize_geometry:
            self.mean_delta_param = nn.Parameter(torch.zeros_like(self.base_means))
            self.scale_delta_param = nn.Parameter(torch.zeros_like(self.base_scales_log))
        else:
            self.register_parameter("mean_delta_param", None)
            self.register_parameter("scale_delta_param", None)
        if self.optimize_quats:
            self.quat_delta_param = nn.Parameter(torch.zeros_like(self.base_quats))
        else:
            self.register_parameter("quat_delta_param", None)

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------
    def appearance_parameters(self) -> list[torch.nn.Parameter]:
        return [self.appearance_rgb]

    def geometry_parameters(self) -> list[torch.nn.Parameter]:
        return self.geometry_mean_scale_parameters() + self.geometry_quat_parameters()

    def geometry_mean_scale_parameters(self) -> list[torch.nn.Parameter]:
        if not self.optimize_geometry:
            return []
        params: list[torch.nn.Parameter] = []
        if self.mean_delta_param is not None:
            params.append(self.mean_delta_param)
        if self.scale_delta_param is not None:
            params.append(self.scale_delta_param)
        return params

    def geometry_quat_parameters(self) -> list[torch.nn.Parameter]:
        if not self.optimize_quats or self.quat_delta_param is None:
            return []
        return [self.quat_delta_param]

    # ------------------------------------------------------------------
    # Geometry / opacity accessors
    # ------------------------------------------------------------------
    def means(self) -> torch.Tensor:
        if not self.optimize_geometry or self.mean_delta_param is None:
            return self.base_means
        return self.base_means + self.max_mean_delta * torch.tanh(self.mean_delta_param)

    def scales_log(self) -> torch.Tensor:
        if not self.optimize_geometry or self.scale_delta_param is None:
            return self.base_scales_log
        return self.base_scales_log + self.max_scale_delta * torch.tanh(self.scale_delta_param)

    def _materialize_quats(self) -> torch.Tensor:
        if not self.optimize_quats or self.quat_delta_param is None:
            return self.base_quats
        return self.base_quats + self.quat_delta()

    def quats(self) -> torch.Tensor:
        return F.normalize(self._materialize_quats(), dim=-1)

    def opacities_logit(self) -> torch.Tensor:
        return self.base_opacities_logit

    def quat_delta(self) -> torch.Tensor:
        if not self.optimize_quats or self.quat_delta_param is None:
            return torch.zeros_like(self.base_quats)
        return self.max_quat_delta * torch.tanh(self.quat_delta_param)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply_parameter_constraints(self) -> None:
        """Keep direct RGB appearance parameters in displayable range."""
        self.appearance_rgb.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------
    # Color / appearance
    # ------------------------------------------------------------------
    def colors(self) -> torch.Tensor:
        """Return the current direct per-Gaussian RGB values."""
        return self.appearance_rgb

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @torch.no_grad()
    def parameter_stats(self) -> dict[str, float]:
        return tensor_range_stats("appearance_rgb", self.colors())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_batch(
        self,
        *,
        c2w: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        clamp: bool = True,
    ) -> torch.Tensor:
        colors = self.colors()

        renders, _, _ = rasterize_gaussians(
            means=self.means(),
            quats=self._materialize_quats(),
            scales_log=self.scales_log(),
            opacities_logit=self.opacities_logit(),
            colors=colors,
            c2w=c2w,
            K=K,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB",
            sh_degree=None,
        )
        rgb = renders[..., :3]
        if clamp:
            rgb = rgb.clamp(0.0, 1.0)
        return rgb

    def render(self, camera: CameraView, *, clamp: bool = True) -> torch.Tensor:
        c2w = camera.c2w.unsqueeze(0)
        K = camera.K.unsqueeze(0)
        rgb_hwc = self.render_batch(
            c2w=c2w,
            K=K,
            width=camera.width,
            height=camera.height,
            clamp=clamp,
        )[0]
        return rgb_hwc.permute(2, 0, 1).contiguous()

    @torch.no_grad()
    def render_current_images(self, cameras: Cameras) -> ViewImages:
        images_by_view = {}
        for camera in cameras:
            images_by_view[camera.view_index] = self.render(camera).detach()
        return ViewImages(images_by_view=images_by_view)

    # ------------------------------------------------------------------
    # Final Gaussian output
    # ------------------------------------------------------------------
    def save_ply(self, path: Path) -> None:
        """Save the materialized scene as an SH0 Gaussian PLY."""
        save_gaussian_ply(
            path,
            splats={
                "means": self.means(),
                "quats": self.quats(),
                "scales": self.scales_log(),
                "opacities": self.opacities_logit(),
                "sh0": rgb_to_sh0(self.colors()).unsqueeze(1),
            },
        )


def build_gaussian_scene(
    ply_path: Path,
    device: torch.device,
    optimize_geometry: bool,
    optimize_quats: bool,
    max_mean_delta: float,
    max_scale_delta: float,
    max_quat_delta: float,
) -> GaussianScene:
    """Build a direct-RGB Gaussian scene from a Gaussian PLY."""
    splats = load_gaussian_ply(ply_path)
    return GaussianScene(
        splats=splats,
        device=device,
        optimize_geometry=optimize_geometry,
        optimize_quats=optimize_quats,
        max_mean_delta=max_mean_delta,
        max_scale_delta=max_scale_delta,
        max_quat_delta=max_quat_delta,
    )
