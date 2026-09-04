"""Deterministic Stable Diffusion VAE codec."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL


class LatentCodec:
    """Encode RGB images with VAE means and decode latents to clamped RGB."""

    def __init__(
        self,
        *,
        vae: AutoencoderKL,
        device: torch.device,
        weight_dtype: torch.dtype,
    ) -> None:
        self._vae = vae
        self._device = device
        self._weight_dtype = weight_dtype

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("VAE RGB input must have shape [B, 3, H, W]")
        normalized = images.to(
            device=self._device,
            dtype=self._weight_dtype,
        ) * 2.0 - 1.0
        encoded = self._vae.encode(normalized)
        latents = encoded.latent_dist.mean * self._vae.config.scaling_factor
        return latents.detach().float()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("VAE latent input must have shape [B, C, H, W]")
        scaled = latents.to(
            device=self._device,
            dtype=self._weight_dtype,
        ) / self._vae.config.scaling_factor
        images = self._vae.decode(scaled).sample
        return (images.float() * 0.5 + 0.5).clamp(0.0, 1.0)
