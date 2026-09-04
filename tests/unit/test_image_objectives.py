from __future__ import annotations

import pytest
import torch

from dresg.models.gs.fitting.image import structural_similarity, total_variation_loss


def test_ssim_is_one_for_identical_images() -> None:
    image = torch.rand((1, 3, 8, 8))

    score = structural_similarity(image, image)

    assert torch.allclose(score, torch.tensor(1.0), atol=1e-6)


def test_total_variation_loss_accepts_chw_and_bchw() -> None:
    image = torch.zeros((3, 2, 3))
    image[:, :, 1:] = 1.0

    chw_loss = total_variation_loss(image)
    bchw_loss = total_variation_loss(image.unsqueeze(0))

    assert torch.allclose(chw_loss, bchw_loss)
    assert torch.allclose(chw_loss, torch.tensor(0.5))


def test_ssim_requires_matching_tensor_contracts() -> None:
    image = torch.rand((1, 3, 8, 8))

    with pytest.raises(ValueError, match="matching shapes"):
        structural_similarity(image, torch.rand((1, 3, 7, 8)))
    with pytest.raises(ValueError, match="share device and dtype"):
        structural_similarity(image, image.double())
    with pytest.raises(ValueError, match="positive odd integer"):
        structural_similarity(image, image, window_size=4)


def test_total_variation_requires_two_spatial_samples() -> None:
    with pytest.raises(ValueError, match="height and width"):
        total_variation_loss(torch.zeros((1, 3, 1, 4)))
