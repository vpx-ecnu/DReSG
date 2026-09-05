from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
    *,
    mode: str = "replicate",
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, height, width = x.shape
    ph = (multiple - height % multiple) % multiple
    pw = (multiple - width % multiple) % multiple
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    if ph or pw:
        x = F.pad(x, pad, mode=mode)
    return x, pad


def unpad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    height, width = x.shape[-2:]
    return x[
        ...,
        top : height - bottom if bottom else height,
        left : width - right if right else width,
    ]


def warp_with_mask(x: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-sample at grid + flow with RAFT's strict interior validity mask."""
    _, _, height, width = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device),
        torch.arange(width, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).float()[None] + flow
    gx = 2.0 * grid[:, 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * grid[:, 1] / max(height - 1, 1) - 1.0
    valid = (gx > -1.0) & (gx < 1.0) & (gy > -1.0) & (gy < 1.0)
    sampled = F.grid_sample(x, torch.stack((gx, gy), dim=-1), mode="bilinear", align_corners=True)
    return sampled, valid[:, None].float()


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    sampled, _ = warp_with_mask(x, flow)
    return sampled
