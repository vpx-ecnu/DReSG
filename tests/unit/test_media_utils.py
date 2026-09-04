from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from dresg.utils.images import chw_to_hwc_u8, image_size, load_rgb_chw01, save_rgb


def test_chw_to_hwc_u8_clamps_and_reorders_channels() -> None:
    image = torch.tensor(
        [
            [[-1.0, 0.0], [0.5, 1.5]],
            [[0.25, 0.5], [0.75, 1.0]],
            [[1.0, 0.75], [0.5, 0.25]],
        ]
    )

    out = chw_to_hwc_u8(image)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert out[0, 0].tolist() == [0, 64, 255]
    assert out[1, 1].tolist() == [255, 255, 64]


def test_save_rgb_creates_parent_and_writes_rgb(tmp_path) -> None:
    path = tmp_path / "nested" / "rgb.png"
    image = torch.zeros((3, 2, 2))
    image[0] = 1.0

    save_rgb(path, image)

    assert path.exists()


def test_load_rgb_chw01_preserves_rgb_values(tmp_path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.array([[[0, 128, 255]]], dtype=np.uint8)).save(path)

    image = load_rgb_chw01(path)

    assert image.shape == (3, 1, 1)
    assert torch.allclose(image[:, 0, 0], torch.tensor([0.0, 128.0 / 255.0, 1.0]))
    assert image_size(path) == (1, 1)


def test_load_rgb_chw01_discards_alpha_channel(tmp_path) -> None:
    path = tmp_path / "rgba.png"
    Image.fromarray(np.array([[[255, 64, 0, 7]]], dtype=np.uint8)).save(path)

    image = load_rgb_chw01(path)

    assert image.shape == (3, 1, 1)
    assert torch.allclose(image[:, 0, 0], torch.tensor([1.0, 64.0 / 255.0, 0.0]))
