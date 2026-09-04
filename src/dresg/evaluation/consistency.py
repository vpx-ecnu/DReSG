"""Train-render temporal consistency metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dresg.utils.flow import pad_to_multiple, unpad, warp, warp_with_mask
from dresg.utils.images import load_rgb_chw01

RaftTransforms = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True, slots=True)
class ConsistencyMetrics:
    lpips: float
    rmse: float


def resolve_gap(spec: str, count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("Consistency view count must be an integer")
    if count < 1:
        raise ValueError("Consistency view count must be positive")
    if not isinstance(spec, str):
        raise TypeError("Consistency gap specification must be a string")
    if spec == "n/2":
        return max(1, count // 2)
    if not spec.isascii() or not spec.isdecimal():
        raise ValueError(f"Invalid consistency gap specification: {spec!r}")
    gap = int(spec)
    if gap < 1 or str(gap) != spec:
        raise ValueError(f"Consistency gap must be a canonical positive integer: {spec!r}")
    return gap


def _load_raft_model(
    device: torch.device,
    *,
    offline_models: bool,
) -> tuple[torch.nn.Module, RaftTransforms]:
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.C_T_SKHT_V2
    if offline_models:
        filename = weights.url.rsplit("/", 1)[-1]
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Offline RAFT checkpoint not found: {checkpoint}")
    model = raft_large(weights=weights, progress=False).to(device).eval()
    return model, weights.transforms()


def _load_lpips_model(
    device: torch.device,
    *,
    offline_models: bool,
) -> torch.nn.Module:
    if offline_models:
        from torchvision.models import AlexNet_Weights

        weights = AlexNet_Weights.IMAGENET1K_V1
        filename = weights.url.rsplit("/", 1)[-1]
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Offline LPIPS AlexNet checkpoint not found: {checkpoint}"
            )
    import lpips

    return lpips.LPIPS(net="alex", spatial=True).to(device).eval()


@torch.no_grad()
def raft_flow(
    model: torch.nn.Module,
    transforms: RaftTransforms,
    a: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x1, x2 = transforms(a.unsqueeze(0).to(device), b.unsqueeze(0).to(device))
    x1, pad = pad_to_multiple(x1, 8)
    x2, _ = pad_to_multiple(x2, 8)
    flow = model(x1, x2)[-1]
    return unpad(flow, pad)


class ConsistencyEvaluator:
    """Own RAFT and spatial LPIPS resources for one evaluation run."""

    def __init__(
        self,
        *,
        raft_model: torch.nn.Module,
        raft_transforms: RaftTransforms,
        lpips_model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self._raft_model = raft_model
        self._raft_transforms = raft_transforms
        self._lpips_model = lpips_model
        self._device = device

    @classmethod
    def load(
        cls,
        device: torch.device,
        *,
        offline_models: bool,
    ) -> ConsistencyEvaluator:
        lpips_model = _load_lpips_model(
            device,
            offline_models=offline_models,
        )
        raft_model, transforms = _load_raft_model(
            device,
            offline_models=offline_models,
        )
        return cls(
            raft_model=raft_model,
            raft_transforms=transforms,
            lpips_model=lpips_model,
            device=device,
        )

    @torch.no_grad()
    def evaluate(
        self,
        content_paths: Sequence[Path],
        stylized_paths: Sequence[Path],
        *,
        gap: int,
        samples: int,
    ) -> ConsistencyMetrics:
        if len(content_paths) != len(stylized_paths):
            raise ValueError(
                "Consistency evaluation requires equal content and stylized view counts: "
                f"content={len(content_paths)}, stylized={len(stylized_paths)}"
            )
        max_start = len(content_paths) - gap - 1
        if max_start < 0:
            raise ValueError(
                f"Consistency gap {gap} has no valid pair among {len(content_paths)} views"
            )
        if samples < 1:
            raise ValueError("Consistency sample count must be positive")

        indices = np.linspace(
            0,
            max_start,
            min(samples, max_start + 1),
        ).round().astype(int)
        lpips_scores = []
        rmse_scores = []
        for idx in indices:
            c0 = load_rgb_chw01(content_paths[idx])
            c1 = load_rgb_chw01(content_paths[idx + gap])
            s0 = load_rgb_chw01(stylized_paths[idx]).to(self._device)
            s1 = load_rgb_chw01(stylized_paths[idx + gap]).to(self._device)
            if s1.shape[-2:] != s0.shape[-2:]:
                s1 = F.interpolate(
                    s1.unsqueeze(0),
                    size=s0.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if c0.shape[-2:] != s0.shape[-2:]:
                c0 = F.interpolate(
                    c0.unsqueeze(0),
                    size=s0.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                c1 = F.interpolate(
                    c1.unsqueeze(0),
                    size=s0.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            fwd = raft_flow(
                self._raft_model,
                self._raft_transforms,
                c0,
                c1,
                self._device,
            )
            bwd = raft_flow(
                self._raft_model,
                self._raft_transforms,
                c1,
                c0,
                self._device,
            )
            s0b = s0.unsqueeze(0)
            s1b = s1.unsqueeze(0)
            warped, inbounds = warp_with_mask(s1b, fwd)
            warped_bwd = warp(bwd, fwd)
            fb_err = (fwd + warped_bwd).square().sum(dim=1, keepdim=True)
            mag = fwd.square().sum(dim=1, keepdim=True) + warped_bwd.square().sum(
                dim=1,
                keepdim=True,
            )
            valid = (fb_err < (0.01 * mag + 0.5)) & (inbounds > 0.5)

            valid_count = valid.float().sum().clamp_min(1.0) * warped.shape[1]
            rmse = (
                ((warped - s0b).square() * valid.float()).sum() / valid_count
            ).sqrt()
            lpips_map = self._lpips_model(
                warped * 2.0 - 1.0,
                s0b * 2.0 - 1.0,
            )
            if lpips_map.ndim != 4:
                raise ValueError(
                    "Consistency evaluation requires a spatial LPIPS model"
                )
            valid_lpips = F.interpolate(
                valid.float(),
                size=lpips_map.shape[-2:],
                mode="nearest",
            )
            lpips_score = (
                (lpips_map * valid_lpips).sum()
                / valid_lpips.sum().clamp_min(1.0)
            )
            lpips_value, rmse_value = torch.stack([lpips_score, rmse]).cpu().tolist()
            lpips_scores.append(lpips_value)
            rmse_scores.append(rmse_value)

        return ConsistencyMetrics(
            lpips=float(np.mean(lpips_scores)),
            rmse=float(np.mean(rmse_scores)),
        )
