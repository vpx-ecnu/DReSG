from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from dresg.data.cameras import Cameras
from dresg.data.images import ViewImages, load_source_view_images


def test_view_images_copies_mapping_and_exposes_sorted_ids() -> None:
    image2 = torch.ones(3, 4, 5)
    source = {
        7: torch.zeros(3, 4, 5),
        2: image2,
    }

    images = ViewImages(source)
    source.clear()

    assert len(images) == 2
    assert images.view_ids == (2, 7)
    assert set(images) == {2, 7}
    assert images[2] is image2
    with pytest.raises(TypeError):
        images[3] = torch.zeros(3, 4, 5)


@pytest.mark.parametrize("view_id", [True, 1.5, "1"])
def test_view_images_rejects_view_id_coercion(view_id: object) -> None:
    with pytest.raises(TypeError, match="must be integers"):
        ViewImages({view_id: torch.zeros(3, 4, 5)})


def test_view_images_rejects_negative_view_id() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ViewImages({-1: torch.zeros(3, 4, 5)})


def test_view_images_rejects_non_rgb_or_mismatched_tensors() -> None:
    with pytest.raises(ValueError, match=r"\[3, H, W\]"):
        ViewImages({0: torch.zeros(1, 4, 5)})
    with pytest.raises(ValueError, match="share one shape"):
        ViewImages(
            {
                0: torch.zeros(3, 4, 5),
                1: torch.zeros(3, 5, 5),
            }
        )
    with pytest.raises(ValueError, match="share one dtype"):
        ViewImages(
            {
                0: torch.zeros(3, 4, 5, dtype=torch.float32),
                1: torch.zeros(3, 4, 5, dtype=torch.float64),
            }
        )


def test_view_images_rejects_noncanonical_tensor_values() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        ViewImages({0: object()})
    with pytest.raises(TypeError, match="floating-point"):
        ViewImages({0: torch.zeros(3, 4, 5, dtype=torch.int64)})
    with pytest.raises(ValueError, match="only finite"):
        ViewImages({0: torch.full((3, 4, 5), float("nan"))})


def test_load_source_view_images_uses_camera_order_and_size(tmp_path) -> None:
    image0 = tmp_path / "view0.png"
    image1 = tmp_path / "view1.png"
    Image.new("RGB", (4, 3), (255, 0, 0)).save(image0)
    Image.new("RGB", (4, 3), (0, 255, 0)).save(image1)
    cameras = Cameras(
        view_indices=(1, 0),
        c2w=torch.eye(4).repeat(2, 1, 1),
        K=torch.eye(3).repeat(2, 1, 1),
        width=4,
        height=3,
    )

    source_images = load_source_view_images(
        SimpleNamespace(image_paths=(image0, image1)),
        cameras,
    )

    assert list(source_images) == [1, 0]
    assert torch.allclose(source_images[1][1], torch.ones((3, 4)))
    assert torch.allclose(source_images[0][0], torch.ones((3, 4)))


def test_view_images_lookup_rejects_boolean_aliases() -> None:
    images = ViewImages({1: torch.zeros(3, 4, 5)})

    with pytest.raises(TypeError, match="View IDs must be integers"):
        images[True]
