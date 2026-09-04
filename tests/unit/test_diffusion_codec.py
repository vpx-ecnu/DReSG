from __future__ import annotations

from types import SimpleNamespace

import torch

from dresg.models.diffusion.latents.codec import LatentCodec


class _VAE:
    config = SimpleNamespace(scaling_factor=0.5)

    def __init__(self) -> None:
        self.encoded = None
        self.decoded = None

    def encode(self, images: torch.Tensor):
        self.encoded = images.clone()
        distribution = SimpleNamespace(mean=images.mean(dim=1, keepdim=True))
        return SimpleNamespace(latent_dist=distribution)

    def decode(self, latents: torch.Tensor):
        self.decoded = latents.clone()
        return SimpleNamespace(sample=latents.repeat(1, 3, 1, 1))


def test_latent_codec_encodes_deterministic_float32_vae_means() -> None:
    vae = _VAE()
    codec = LatentCodec(
        vae=vae,
        device=torch.device("cpu"),
        weight_dtype=torch.float64,
    )
    images = torch.full((1, 3, 2, 2), 0.75, requires_grad=True)

    latents = codec.encode(images)

    assert vae.encoded.dtype == torch.float64
    assert torch.allclose(vae.encoded, torch.full((1, 3, 2, 2), 0.5, dtype=torch.float64))
    assert latents.dtype == torch.float32
    assert not latents.requires_grad
    assert torch.allclose(latents, torch.full((1, 1, 2, 2), 0.25))


def test_latent_codec_decodes_scaling_factor_and_clamps_rgb() -> None:
    vae = _VAE()
    codec = LatentCodec(
        vae=vae,
        device=torch.device("cpu"),
        weight_dtype=torch.float32,
    )

    images = codec.decode(torch.full((1, 1, 2, 2), 0.5))

    assert torch.allclose(vae.decoded, torch.ones((1, 1, 2, 2)))
    assert images.shape == (1, 3, 2, 2)
    assert torch.all(images == 1.0)


def test_latent_codec_rejects_invalid_rgb_shape() -> None:
    codec = LatentCodec(
        vae=_VAE(),
        device=torch.device("cpu"),
        weight_dtype=torch.float32,
    )

    try:
        codec.encode(torch.zeros(3, 2, 2))
    except ValueError as error:
        assert "[B, 3, H, W]" in str(error)
    else:
        raise AssertionError("Invalid RGB shape was accepted")
