"""Image-space loss primitives used by Gaussian fitting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batched_image(image: torch.Tensor, *, name: str) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"{name} must have shape [C, H, W] or [B, C, H, W]")
    if any(size < 1 for size in image.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if not image.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    return image


def structural_similarity(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    image_a = _as_batched_image(image_a, name="image_a")
    image_b = _as_batched_image(image_b, name="image_b")
    if image_a.shape != image_b.shape:
        raise ValueError("SSIM images must have matching shapes")
    if image_a.device != image_b.device or image_a.dtype != image_b.dtype:
        raise ValueError("SSIM images must share device and dtype")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size <= 0
        or window_size % 2 == 0
    ):
        raise ValueError("SSIM window_size must be a positive odd integer")

    channels = image_a.shape[1]
    window = torch.ones(
        (channels, 1, window_size, window_size),
        device=image_a.device,
        dtype=image_a.dtype,
    )
    window = window / float(window_size * window_size)
    mu_a = F.conv2d(image_a, window, padding=window_size // 2, groups=channels)
    mu_b = F.conv2d(image_b, window, padding=window_size // 2, groups=channels)
    mu_a2 = mu_a.square()
    mu_b2 = mu_b.square()
    mu_ab = mu_a * mu_b
    sigma_a = (
        F.conv2d(image_a * image_a, window, padding=window_size // 2, groups=channels)
        - mu_a2
    )
    sigma_b = (
        F.conv2d(image_b * image_b, window, padding=window_size // 2, groups=channels)
        - mu_b2
    )
    sigma_ab = (
        F.conv2d(image_a * image_b, window, padding=window_size // 2, groups=channels)
        - mu_ab
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a + sigma_b + c2)
    return (numerator / denominator).mean()


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    image = _as_batched_image(image, name="image")
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("Total variation requires image height and width of at least 2")
    return (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean() + (
        image[:, :, 1:, :] - image[:, :, :-1, :]
    ).abs().mean()
