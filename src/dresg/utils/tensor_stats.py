from __future__ import annotations

from statistics import fmean

import torch


def tensor_range_stats(prefix: str, value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_under0_frac": 0.0,
            f"{prefix}_over1_frac": 0.0,
        }
    minimum, maximum, under, over = torch.stack(
        [
            flat.min(),
            flat.max(),
            (flat < 0.0).float().mean(),
            (flat > 1.0).float().mean(),
        ]
    ).cpu().tolist()
    return {
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_under0_frac": under,
        f"{prefix}_over1_frac": over,
    }


def image_color_stats(prefix: str, image: torch.Tensor) -> dict[str, float]:
    image = image.detach().float().clamp(0.0, 1.0)
    if image.ndim == 4:
        channels = image.shape[1]
        flat = image.permute(0, 2, 3, 1).reshape(-1, channels)
    else:
        channels = image.shape[0]
        flat = image.permute(1, 2, 0).reshape(-1, channels)
    rgb = flat[:, :3]
    max_rgb = rgb.max(dim=1).values
    min_rgb = rgb.min(dim=1).values
    chroma = max_rgb - min_rgb
    saturation = torch.where(max_rgb > 1e-6, chroma / max_rgb.clamp_min(1e-6), torch.zeros_like(max_rgb))
    saturation_mean, value_mean, chroma_mean, clip_hi, clip_lo = torch.stack(
        [
            saturation.mean(),
            max_rgb.mean(),
            chroma.mean(),
            (rgb > 0.98).float().mean(),
            (rgb < 0.02).float().mean(),
        ]
    ).cpu().tolist()
    return {
        f"{prefix}_saturation_mean": saturation_mean,
        f"{prefix}_value_mean": value_mean,
        f"{prefix}_chroma_mean": chroma_mean,
        f"{prefix}_clip_hi_frac": clip_hi,
        f"{prefix}_clip_lo_frac": clip_lo,
    }


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {
        key: fmean(item[key] for item in items)
        for key in sorted(items[0])
    }
