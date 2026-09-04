from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from dresg.utils.images import load_rgb_image

if TYPE_CHECKING:
    from dresg.data.cameras import Cameras
    from dresg.data.colmap import ColmapScene


def _require_view_id(view_id: int) -> int:
    if isinstance(view_id, bool) or not isinstance(view_id, int):
        raise TypeError("View IDs must be integers")
    if view_id < 0:
        raise ValueError("View IDs must be non-negative")
    return view_id


class ViewImages(Mapping[int, torch.Tensor]):
    """Immutable mapping of canonical view IDs to finite RGB tensors."""

    def __init__(self, images_by_view: Mapping[int, torch.Tensor]) -> None:
        images = {
            _require_view_id(view_id): image
            for view_id, image in images_by_view.items()
        }
        if images:
            first = next(iter(images.values()))
            if not isinstance(first, torch.Tensor):
                raise TypeError("View images must be torch.Tensor instances")
            if first.ndim != 3 or first.shape[0] != 3:
                raise ValueError("View images must have shape [3, H, W]")
            if first.shape[1] < 1 or first.shape[2] < 1:
                raise ValueError("View image dimensions must be positive")
            if not first.is_floating_point():
                raise TypeError("View images must use a floating-point dtype")
            for view_id, image in images.items():
                if not isinstance(image, torch.Tensor):
                    raise TypeError(f"View {view_id} image must be a torch.Tensor")
                if image.ndim != 3 or image.shape[0] != 3:
                    raise ValueError(f"View {view_id} image must have shape [3, H, W]")
                if image.shape != first.shape:
                    raise ValueError("All view images must share one shape")
                if image.device != first.device:
                    raise ValueError("All view images must share one device")
                if image.dtype != first.dtype:
                    raise ValueError("All view images must share one dtype")
            finite = torch.stack(
                [torch.isfinite(image).all() for image in images.values()]
            ).all()
            if not finite:
                raise ValueError("View images must contain only finite values")
        self._images = MappingProxyType(images)

    def __getitem__(self, view_id: int) -> torch.Tensor:
        return self._images[_require_view_id(view_id)]

    def __iter__(self) -> Iterator[int]:
        return iter(self._images)

    def __len__(self) -> int:
        return len(self._images)

    @property
    def view_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._images))


def load_source_view_images(
    source: ColmapScene,
    cameras: Cameras,
) -> ViewImages:
    """Load active source pixels at the canonical camera render size."""
    return ViewImages(
        images_by_view={
            view_id: load_rgb_image(
                source.image_paths[view_id],
                device=cameras.c2w.device,
                width=cameras.width,
                height=cameras.height,
            )
            for view_id in cameras.view_indices
        }
    )
