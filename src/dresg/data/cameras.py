from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from dresg.data.colmap import ColmapScene


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_integer(value: int, *, name: str) -> int:
    number = _require_nonnegative_integer(value, name=name)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True)
class CameraView:
    view_index: int
    c2w: torch.Tensor
    K: torch.Tensor
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.view_index, name="view_index")
        if not isinstance(self.c2w, torch.Tensor) or not isinstance(self.K, torch.Tensor):
            raise TypeError("c2w and K must be torch.Tensor instances")
        if self.c2w.shape != (4, 4):
            raise ValueError("c2w must have shape [4, 4]")
        if self.K.shape != (3, 3):
            raise ValueError("K must have shape [3, 3]")
        if self.c2w.device != self.K.device:
            raise ValueError("c2w and K must share one device")
        if self.c2w.dtype != self.K.dtype:
            raise ValueError("c2w and K must share one dtype")
        if not self.c2w.is_floating_point() or not self.K.is_floating_point():
            raise TypeError("c2w and K must use floating-point dtypes")
        _require_positive_integer(self.width, name="Camera width")
        _require_positive_integer(self.height, name="Camera height")


@dataclass(frozen=True)
class Cameras:
    """Validated camera tensors with CPU-resident canonical view IDs."""

    view_indices: tuple[int, ...]
    c2w: torch.Tensor
    K: torch.Tensor
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.view_indices, tuple):
            raise TypeError("view_indices must be a tuple")
        for view_index in self.view_indices:
            _require_nonnegative_integer(view_index, name="view_indices entries")
        if len(set(self.view_indices)) != len(self.view_indices):
            raise ValueError("view_indices must be unique")
        if not isinstance(self.c2w, torch.Tensor) or not isinstance(self.K, torch.Tensor):
            raise TypeError("c2w and K must be torch.Tensor instances")
        if self.c2w.ndim != 3 or self.c2w.shape[1:] != (4, 4):
            raise ValueError("c2w must have shape [N, 4, 4]")
        if self.K.ndim != 3 or self.K.shape[1:] != (3, 3):
            raise ValueError("K must have shape [N, 3, 3]")
        if not (len(self.view_indices) == self.c2w.shape[0] == self.K.shape[0]):
            raise ValueError("view_indices, c2w, and K must share the same leading dimension")
        if self.c2w.device != self.K.device:
            raise ValueError("c2w and K must share one device")
        if self.c2w.dtype != self.K.dtype:
            raise ValueError("c2w and K must share one dtype")
        if not self.c2w.is_floating_point() or not self.K.is_floating_point():
            raise TypeError("c2w and K must use floating-point dtypes")
        if not torch.isfinite(self.c2w).all() or not torch.isfinite(self.K).all():
            raise ValueError("Camera tensors must contain only finite values")
        _require_positive_integer(self.width, name="Camera width")
        _require_positive_integer(self.height, name="Camera height")

    def __len__(self) -> int:
        return len(self.view_indices)

    def __iter__(self) -> Iterator[CameraView]:
        for idx in range(len(self)):
            yield self.view(idx)

    def view(self, idx: int) -> CameraView:
        index = _require_nonnegative_integer(idx, name="Camera position")
        if index >= len(self):
            raise IndexError(f"Camera position {index} is outside 0..{len(self) - 1}")
        return CameraView(
            view_index=self.view_indices[index],
            c2w=self.c2w[index],
            K=self.K[index],
            width=self.width,
            height=self.height,
        )

    @classmethod
    def empty(cls, *, device: torch.device, width: int, height: int) -> Cameras:
        return cls(
            view_indices=(),
            c2w=torch.empty((0, 4, 4), device=device),
            K=torch.empty((0, 3, 3), device=device),
            width=width,
            height=height,
        )


def scaled_intrinsics(
    K: torch.Tensor,
    width: int,
    height: int,
    scale: float,
) -> tuple[torch.Tensor, int, int]:
    """Scale intrinsics using the realized integer output dimensions."""
    if K.shape[-2:] != (3, 3):
        raise ValueError("K must end with shape [3, 3]")
    width = _require_positive_integer(width, name="Camera width")
    height = _require_positive_integer(height, name="Camera height")
    if isinstance(scale, bool):
        raise TypeError("Camera scale must be numeric")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Camera scale must be finite and positive")
    if abs(scale - 1.0) < 1e-8:
        return K, width, height
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    sx = new_w / width
    sy = new_h / height
    K2 = K.clone()
    K2[..., 0, 0] *= sx
    K2[..., 1, 1] *= sy
    K2[..., 0, 2] *= sx
    K2[..., 1, 2] *= sy
    return K2, new_w, new_h


def build_scaled_cameras(
    *,
    source: ColmapScene,
    view_ids: Sequence[int],
    device: torch.device,
    render_scale: float,
    label: str,
    reference_width: int | None = None,
    reference_height: int | None = None,
) -> Cameras:
    if (reference_width is None) != (reference_height is None):
        raise ValueError("reference_width and reference_height must be provided together")
    if reference_width is not None:
        reference_width = _require_positive_integer(reference_width, name="reference_width")
        reference_height = _require_positive_integer(reference_height, name="reference_height")

    view_indices = tuple(
        _require_nonnegative_integer(view_id, name="view_ids entries")
        for view_id in view_ids
    )
    if len(set(view_indices)) != len(view_indices):
        raise ValueError("view_ids must be unique")
    if not view_indices:
        if reference_width is None or reference_height is None:
            raise ValueError("Cannot build empty Cameras without a reference resolution")
        return Cameras.empty(
            device=device,
            width=reference_width,
            height=reference_height,
        )

    K_list: list[torch.Tensor] = []
    for view_index in view_indices:
        source.validate_view_index(view_index)
        camera_id = source.camera_ids[view_index]
        width, height = source.image_sizes_by_camera[camera_id]
        K = torch.from_numpy(
            np.array(source.intrinsics_by_camera[camera_id], copy=True)
        ).to(dtype=torch.float32)
        K, width, height = scaled_intrinsics(
            K,
            width,
            height,
            render_scale,
        )
        if reference_width is None:
            reference_width, reference_height = width, height
        assert reference_height is not None
        if (width, height) != (reference_width, reference_height):
            raise ValueError(f"All {label} camera resolutions must match after scaling")
        K_list.append(K)

    assert reference_width is not None and reference_height is not None
    c2w_array = np.ascontiguousarray(source.camtoworlds[list(view_indices)])
    c2w = torch.from_numpy(c2w_array).to(device=device, dtype=torch.float32)
    K = torch.stack(K_list, dim=0).to(device=device)
    return Cameras(
        view_indices=view_indices,
        c2w=c2w,
        K=K,
        width=reference_width,
        height=reference_height,
    )
