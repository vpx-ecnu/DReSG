"""Strict PLY I/O for SH0 Gaussian scenes."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

SplatTensors = dict[str, torch.Tensor]

_SPLAT_SHAPES: dict[str, tuple[int, ...]] = {
    "means": (3,),
    "quats": (4,),
    "scales": (3,),
    "opacities": (),
    "sh0": (1, 3),
}
_REQUIRED_VERTEX_FIELDS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
_NORMAL_FIELDS = ("nx", "ny", "nz")
_OUTPUT_VERTEX_FIELDS = (
    "x",
    "y",
    "z",
    *_NORMAL_FIELDS,
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _require_ply_path(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError("Gaussian PLY path must be a pathlib.Path")
    if path.suffix != ".ply":
        raise ValueError(f"Gaussian PLY path must use the .ply suffix: {path}")


def validate_gaussian_splats(splats: Mapping[str, object]) -> SplatTensors:
    """Validate the one in-memory SH0 splat layout used by DReSG."""
    if any(not isinstance(name, str) for name in splats):
        raise TypeError("Splat keys must be strings")
    expected = set(_SPLAT_SHAPES)
    actual = set(splats)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Invalid splat keys: missing={missing}, extra={extra}")

    tensors: SplatTensors = {}
    count: int | None = None
    for name, trailing_shape in _SPLAT_SHAPES.items():
        value = splats[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"splats.{name} must be a torch.Tensor")
        expected_ndim = len(trailing_shape) + 1
        if value.ndim != expected_ndim or tuple(value.shape[1:]) != trailing_shape:
            dimensions = "".join(f", {size}" for size in trailing_shape)
            raise ValueError(
                f"splats.{name} must have shape [N{dimensions}], "
                f"got {tuple(value.shape)}"
            )
        if count is None:
            count = value.shape[0]
            if count < 1:
                raise ValueError("A Gaussian scene must contain at least one splat")
        elif value.shape[0] != count:
            raise ValueError(f"splats.{name} has {value.shape[0]} entries; expected {count}")
        if not value.is_floating_point():
            raise TypeError(f"splats.{name} must use a floating-point dtype")
        if value.dtype != torch.float32:
            raise TypeError(f"splats.{name} must use torch.float32")
        if not torch.isfinite(value).all():
            raise ValueError(f"splats.{name} contains non-finite values")
        if name == "quats" and torch.any(
            torch.linalg.vector_norm(value.float(), dim=-1) <= 1.0e-12
        ):
            raise ValueError("splats.quats must have non-zero norm")
        tensors[name] = value
    return tensors


def _validate_vertex_fields(data: np.ndarray) -> None:
    field_names = set(data.dtype.names or ())
    missing = sorted(set(_REQUIRED_VERTEX_FIELDS) - field_names)
    if missing:
        raise ValueError(f"PLY vertex data is missing required fields: {missing}")


def _retain_finite_gaussians(data: np.ndarray, *, path: Path) -> np.ndarray:
    finite = np.ones(len(data), dtype=bool)
    for name in _REQUIRED_VERTEX_FIELDS:
        finite &= np.isfinite(data[name])
    retained_count = int(finite.sum())
    if retained_count < 1:
        raise ValueError("PLY vertex data contains no Gaussian with finite retained fields")
    removed_count = len(data) - retained_count
    if removed_count:
        warnings.warn(
            f"Discarded {removed_count} Gaussian(s) with non-finite retained "
            f"fields from {path}",
            stacklevel=3,
        )
        return data[finite]
    return data


def _read_vertex_data(path: Path) -> np.ndarray:
    _require_ply_path(path)
    ply = PlyData.read(str(path))
    try:
        data = ply["vertex"].data
    except KeyError as error:
        raise ValueError(f"PLY file has no vertex element: {path}") from error
    if len(data) < 1:
        raise ValueError("PLY vertex data must contain at least one Gaussian")
    _validate_vertex_fields(data)
    return _retain_finite_gaussians(data, path=path)


def _stack_fields(data: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([data[name] for name in names], axis=1),
        dtype=np.float32,
    )


def load_gaussian_ply(path: Path) -> SplatTensors:
    """Load Graphdeco/Fast-PGSR PLY fields and retain degree-zero appearance."""
    data = _read_vertex_data(path)
    splats = {
        "means": torch.from_numpy(_stack_fields(data, ("x", "y", "z"))),
        "quats": torch.from_numpy(
            _stack_fields(data, ("rot_0", "rot_1", "rot_2", "rot_3"))
        ),
        "scales": torch.from_numpy(
            _stack_fields(data, ("scale_0", "scale_1", "scale_2"))
        ),
        "opacities": torch.from_numpy(
            np.ascontiguousarray(data["opacity"], dtype=np.float32)
        ),
        "sh0": torch.from_numpy(
            _stack_fields(data, ("f_dc_0", "f_dc_1", "f_dc_2"))[:, None, :]
        ),
    }
    return validate_gaussian_splats(splats)


def save_gaussian_ply(
    path: Path,
    *,
    splats: Mapping[str, object],
) -> None:
    """Atomically save a binary little-endian SH0 Gaussian PLY."""
    _require_ply_path(path)
    tensors = validate_gaussian_splats(splats)
    arrays = {
        name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in tensors.items()
    }
    count = arrays["means"].shape[0]
    vertices = np.zeros(
        count,
        dtype=[(name, "<f4") for name in _OUTPUT_VERTEX_FIELDS],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = arrays["means"][:, axis]
    for channel, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        vertices[name] = arrays["sh0"][:, 0, channel]
    vertices["opacity"] = arrays["opacities"]
    for axis in range(3):
        vertices[f"scale_{axis}"] = arrays["scales"][:, axis]
    for component in range(4):
        vertices[f"rot_{component}"] = arrays["quats"][:, component]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        PlyData(
            [PlyElement.describe(vertices, "vertex")],
            text=False,
            byte_order="<",
        ).write(str(temporary_path))
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
