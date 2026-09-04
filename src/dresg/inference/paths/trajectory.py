"""Validated camera trajectories and their construction."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from dresg.data.colmap import ColmapScene
from dresg.inference.paths.geometry import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
    homogeneous_poses,
    validate_camera_frames,
)
from dresg.inference.paths.llff import (
    build_llff_spiral,
    load_forward_facing_bounds,
)
from dresg.inference.paths.tnt import build_tnt_ellipse_path

GenerationValue = str | bool | int | float
_COMMON_GENERATION_FIELDS = {
    "trajectory",
    "camera_source",
    "n_frames",
}
_TRAJECTORY_FIELDS = {
    "builtin_spiral": {"llff_radius_scale"},
    "ellipse_z": set(),
    "interpolated": set(),
    "llff_spiral": {
        "coord_mode",
        "llff_radius_scale",
        "centered_llff_radius",
    },
    "tnt_ellipse": {"ellipse_scale"},
}
_TRAJECTORIES = set(_TRAJECTORY_FIELDS)
_CAMERA_SOURCE_FIELDS = {
    "all": set(),
    "interior": set(),
    "train_split": {"test_every"},
}
_CAMERA_SOURCES = set(_CAMERA_SOURCE_FIELDS)
_COORD_MODES = {"flip_y", "flip_yz", "flip_z", "none"}
_BOUNDS_TRAJECTORIES = {"builtin_spiral", "llff_spiral"}


def _positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _positive_scale(value: float | None, *, name: str) -> None:
    if not isinstance(value, float):
        raise TypeError(f"{name} must be a float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_absent(
    value: str | bool | int | float | None,
    *,
    name: str,
    context: str,
) -> None:
    if value is not None:
        raise ValueError(f"{name} does not apply to {context}")


@dataclass(frozen=True, slots=True)
class VideoPathRequest:
    """Strict trajectory-specific values for one path construction."""

    trajectory: str
    camera_source: str
    test_every: int | None
    n_frames: int
    coord_mode: str | None
    llff_radius_scale: float | None
    ellipse_scale: float | None
    centered_llff_radius: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, str):
            raise TypeError("trajectory must be a string")
        if self.trajectory not in _TRAJECTORIES:
            raise ValueError(f"Unsupported video trajectory: {self.trajectory}")
        if not isinstance(self.camera_source, str):
            raise TypeError("camera_source must be a string")
        if self.camera_source not in _CAMERA_SOURCES:
            raise ValueError(f"Unsupported video camera source: {self.camera_source}")
        if self.camera_source == "train_split":
            if self.test_every is None:
                raise ValueError("test_every is required for camera source train_split")
            _positive_integer(self.test_every, name="test_every")
        else:
            _require_absent(
                self.test_every,
                name="test_every",
                context=f"camera source {self.camera_source}",
            )
        _positive_integer(self.n_frames, name="n_frames")
        trajectory_context = f"trajectory {self.trajectory}"

        if self.trajectory == "llff_spiral":
            if not isinstance(self.coord_mode, str):
                raise TypeError("coord_mode must be a string for llff_spiral")
            if self.coord_mode not in _COORD_MODES:
                raise ValueError(f"Unsupported coord mode: {self.coord_mode}")
            _positive_scale(self.llff_radius_scale, name="llff_radius_scale")
            if not isinstance(self.centered_llff_radius, bool):
                raise TypeError("centered_llff_radius must be a boolean")
            _require_absent(
                self.ellipse_scale,
                name="ellipse_scale",
                context=trajectory_context,
            )
        elif self.trajectory == "builtin_spiral":
            _positive_scale(self.llff_radius_scale, name="llff_radius_scale")
            _require_absent(
                self.coord_mode,
                name="coord_mode",
                context=trajectory_context,
            )
            _require_absent(
                self.ellipse_scale,
                name="ellipse_scale",
                context=trajectory_context,
            )
            _require_absent(
                self.centered_llff_radius,
                name="centered_llff_radius",
                context=trajectory_context,
            )
        elif self.trajectory == "tnt_ellipse":
            _positive_scale(self.ellipse_scale, name="ellipse_scale")
            _require_absent(
                self.coord_mode,
                name="coord_mode",
                context=trajectory_context,
            )
            _require_absent(
                self.llff_radius_scale,
                name="llff_radius_scale",
                context=trajectory_context,
            )
            _require_absent(
                self.centered_llff_radius,
                name="centered_llff_radius",
                context=trajectory_context,
            )
        else:
            for name in (
                "coord_mode",
                "llff_radius_scale",
                "ellipse_scale",
                "centered_llff_radius",
            ):
                _require_absent(
                    getattr(self, name),
                    name=name,
                    context=trajectory_context,
                )

    @classmethod
    def from_metadata(
        cls,
        generation: Mapping[str, GenerationValue],
    ) -> VideoPathRequest:
        if not isinstance(generation, Mapping):
            raise TypeError("Video path generation must be a mapping")
        if any(not isinstance(name, str) for name in generation):
            raise TypeError("Video path generation keys must be strings")
        trajectory = generation.get("trajectory")
        if not isinstance(trajectory, str):
            raise TypeError("Video path generation trajectory must be a string")
        if trajectory not in _TRAJECTORIES:
            raise ValueError(
                f"Unsupported video path generation trajectory: {trajectory}"
            )
        camera_source = generation.get("camera_source")
        if not isinstance(camera_source, str):
            raise TypeError("Video path generation camera_source must be a string")
        if camera_source not in _CAMERA_SOURCES:
            raise ValueError(
                f"Unsupported video path generation camera_source: {camera_source}"
            )
        expected_fields = (
            _COMMON_GENERATION_FIELDS
            | _TRAJECTORY_FIELDS[trajectory]
            | _CAMERA_SOURCE_FIELDS[camera_source]
        )
        actual_fields = set(generation)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            raise ValueError(
                "Invalid video path generation fields: "
                f"missing={missing}, extra={extra}"
            )
        return cls(
            trajectory=trajectory,
            camera_source=camera_source,
            test_every=cast(int | None, generation.get("test_every")),
            n_frames=cast(int, generation["n_frames"]),
            coord_mode=cast(str | None, generation.get("coord_mode")),
            llff_radius_scale=cast(
                float | None,
                generation.get("llff_radius_scale"),
            ),
            ellipse_scale=cast(
                float | None,
                generation.get("ellipse_scale"),
            ),
            centered_llff_radius=cast(
                bool | None,
                generation.get("centered_llff_radius"),
            ),
        )

    def metadata(self) -> dict[str, GenerationValue]:
        values: dict[str, GenerationValue] = {
            "trajectory": self.trajectory,
            "camera_source": self.camera_source,
            "n_frames": self.n_frames,
        }
        if self.camera_source == "train_split":
            assert self.test_every is not None
            values["test_every"] = self.test_every
        if self.trajectory == "llff_spiral":
            assert self.coord_mode is not None
            assert self.llff_radius_scale is not None
            assert self.centered_llff_radius is not None
            values.update(
                coord_mode=self.coord_mode,
                llff_radius_scale=self.llff_radius_scale,
                centered_llff_radius=self.centered_llff_radius,
            )
        elif self.trajectory == "builtin_spiral":
            assert self.llff_radius_scale is not None
            values["llff_radius_scale"] = self.llff_radius_scale
        elif self.trajectory == "tnt_ellipse":
            assert self.ellipse_scale is not None
            values["ellipse_scale"] = self.ellipse_scale
        return values


def _canonical_array_bytes(array: np.ndarray, *, dtype: str) -> bytes:
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    shape = np.asarray(canonical.shape, dtype="<i8")
    return shape.tobytes() + canonical.tobytes()


def scene_fingerprint(source: ColmapScene, *, trajectory: str) -> str:
    """Hash reconstruction coordinates and trajectory-dependent scene inputs."""
    if trajectory not in _TRAJECTORIES:
        raise ValueError(f"Unsupported video trajectory: {trajectory}")
    digest = hashlib.sha256()
    digest.update(b"dresg-video-path-scene-v3\0")
    digest.update(str(source.factor).encode("ascii") + b"\0")
    for image_name in source.image_names:
        encoded = image_name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(_canonical_array_bytes(source.camtoworlds, dtype="<f8"))
    digest.update(
        _canonical_array_bytes(np.asarray(source.camera_ids), dtype="<i8")
    )
    for camera_id in sorted(set(source.camera_ids)):
        digest.update(camera_id.to_bytes(8, "little", signed=True))
        digest.update(
            _canonical_array_bytes(
                source.intrinsics_by_camera[camera_id],
                dtype="<f8",
            )
        )
        digest.update(
            _canonical_array_bytes(
                np.asarray(source.image_sizes_by_camera[camera_id]),
                dtype="<i8",
            )
        )
    if trajectory in _BOUNDS_TRAJECTORIES:
        digest.update(b"forward-facing-bounds-v1\0")
        digest.update(
            _canonical_array_bytes(
                load_forward_facing_bounds(source),
                dtype="<f8",
            )
        )
    return digest.hexdigest()


def _readonly_float32(array: np.ndarray) -> NDArray[np.float32]:
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class VideoPath:
    """Validated fixed camera trajectory and canonical source calibration."""

    c2w: NDArray[np.float32]
    K: NDArray[np.float32]
    width: int
    height: int
    scene_fingerprint: str
    generation: Mapping[str, GenerationValue]

    def __post_init__(self) -> None:
        if not isinstance(self.c2w, np.ndarray) or self.c2w.dtype != np.float32:
            raise TypeError("Video path c2w must be a float32 NumPy array")
        if self.c2w.ndim != 3 or self.c2w.shape[0] < 1 or self.c2w.shape[1:] != (4, 4):
            raise ValueError("Video path c2w must have shape [F, 4, 4]")
        if not np.isfinite(self.c2w).all():
            raise ValueError("Video path c2w must contain only finite values")
        expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if not np.allclose(self.c2w[:, 3], expected_bottom, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "Video path c2w must contain homogeneous camera matrices"
            )
        validate_camera_frames(self.c2w[:, :3, :3])

        if not isinstance(self.K, np.ndarray) or self.K.dtype != np.float32:
            raise TypeError("Video path K must be a float32 NumPy array")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("Video path K must be one finite [3, 3] matrix")
        if self.K[0, 0] <= 0.0 or self.K[1, 1] <= 0.0:
            raise ValueError("Video path focal lengths must be positive")
        if not np.allclose(
            self.K,
            np.array(
                [
                    [self.K[0, 0], 0.0, self.K[0, 2]],
                    [0.0, self.K[1, 1], self.K[1, 2]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("Video path K must use canonical pinhole form")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError("Video path width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height < 1:
            raise ValueError("Video path height must be a positive integer")
        if (
            not isinstance(self.scene_fingerprint, str)
            or len(self.scene_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.scene_fingerprint
            )
        ):
            raise ValueError(
                "Video path scene_fingerprint must be a SHA-256 hex digest"
            )

        request = VideoPathRequest.from_metadata(self.generation)
        if request.n_frames != self.c2w.shape[0]:
            raise ValueError(
                "Video path generation n_frames must match the pose count"
            )
        object.__setattr__(self, "c2w", _readonly_float32(self.c2w))
        object.__setattr__(self, "K", _readonly_float32(self.K))
        object.__setattr__(
            self,
            "generation",
            MappingProxyType(request.metadata()),
        )

    @property
    def frame_count(self) -> int:
        return self.c2w.shape[0]

    @property
    def trajectory(self) -> str:
        return cast(str, self.generation["trajectory"])


def select_video_source_indices(
    source: ColmapScene,
    *,
    camera_source: str,
    test_every: int | None,
) -> np.ndarray:
    if not isinstance(camera_source, str):
        raise TypeError("camera_source must be a string")
    if camera_source not in _CAMERA_SOURCES:
        raise ValueError(f"Unsupported video camera source: {camera_source}")
    indices = np.arange(len(source))
    if camera_source == "train_split":
        if test_every is None:
            raise ValueError("test_every is required for camera source train_split")
        _positive_integer(test_every, name="test_every")
        return indices[indices % test_every != 0]
    if test_every is not None:
        raise ValueError(f"test_every does not apply to camera source {camera_source}")
    if camera_source == "all":
        return indices
    if camera_source == "interior":
        return indices[5:-5]
    raise ValueError(f"Unsupported video camera source: {camera_source}")


def video_camera_parameters(
    source: ColmapScene,
    *,
    camera_source: str,
    test_every: int | None,
) -> tuple[np.ndarray, int, int]:
    indices = select_video_source_indices(
        source,
        camera_source=camera_source,
        test_every=test_every,
    )
    if indices.size == 0:
        raise ValueError("Video camera selection produced no source cameras")
    camera_ids = list(
        dict.fromkeys(source.camera_ids[index] for index in indices)
    )
    first_id = camera_ids[0]
    first_k = np.asarray(source.intrinsics_by_camera[first_id])
    first_size = source.image_sizes_by_camera[first_id]
    for camera_id in camera_ids[1:]:
        camera_k = np.asarray(source.intrinsics_by_camera[camera_id])
        camera_size = source.image_sizes_by_camera[camera_id]
        if camera_size != first_size or not np.allclose(
            camera_k,
            first_k,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                "Video rendering requires camera IDs with matching intrinsics "
                "and image dimensions"
            )
    width, height = first_size
    return first_k.copy(), width, height


def camera_scene_scale(source: ColmapScene) -> float:
    """Measure scene scale from canonical camera centers."""
    camera_locations = source.camtoworlds[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    scale = float(np.linalg.norm(camera_locations - scene_center, axis=1).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Camera trajectory requires a finite positive scene scale")
    return scale


def _validate_built_video_poses(
    poses: np.ndarray,
    *,
    n_frames: int,
) -> np.ndarray:
    poses = homogeneous_poses(np.asarray(poses))
    expected_shape = (n_frames, 4, 4)
    if poses.shape != expected_shape:
        raise ValueError(
            f"Video path must have shape {expected_shape}, got {poses.shape}"
        )
    if not np.issubdtype(poses.dtype, np.floating):
        raise TypeError("Video path poses must use a floating-point dtype")
    if not np.isfinite(poses).all():
        raise ValueError("Video path poses must contain only finite values")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=poses.dtype)
    if not np.allclose(poses[:, 3], expected_bottom, rtol=0.0, atol=1.0e-8):
        raise ValueError("Video path poses must use homogeneous camera matrices")
    validate_camera_frames(poses[:, :3, :3])
    return np.ascontiguousarray(poses)


def build_scene_video_poses(
    source: ColmapScene,
    request: VideoPathRequest,
) -> np.ndarray:
    """Build the exact requested number of canonical camera-to-world poses."""
    indices = select_video_source_indices(
        source,
        camera_source=request.camera_source,
        test_every=request.test_every,
    )
    if indices.size == 0:
        raise ValueError("Video camera selection produced no source cameras")
    camtoworlds = source.camtoworlds[indices]
    if request.trajectory == "interpolated":
        if len(camtoworlds) == 1:
            poses = np.repeat(camtoworlds[:, :3, :4], request.n_frames, axis=0)
        else:
            poses = generate_interpolated_path(camtoworlds, request.n_frames)
    elif request.trajectory == "ellipse_z":
        poses = generate_ellipse_path_z(
            camtoworlds,
            n_frames=request.n_frames,
            height=camtoworlds[:, 2, 3].mean(),
        )
    elif request.trajectory == "tnt_ellipse":
        poses = build_tnt_ellipse_path(
            camtoworlds,
            n_frames=request.n_frames,
            ellipse_scale=request.ellipse_scale,
        )
    elif request.trajectory == "builtin_spiral":
        poses = generate_spiral_path(
            camtoworlds,
            bounds=load_forward_facing_bounds(source)
            * camera_scene_scale(source)
            * 1.1,
            n_frames=request.n_frames,
            spiral_scale_r=request.llff_radius_scale,
        )
    elif request.trajectory == "llff_spiral":
        poses = build_llff_spiral(
            source,
            camtoworlds,
            coord_mode=request.coord_mode,
            centered_radius=request.centered_llff_radius,
            radius_scale=request.llff_radius_scale,
            n_frames=request.n_frames,
        )
    else:
        raise AssertionError(f"Unhandled video trajectory: {request.trajectory}")
    return _validate_built_video_poses(poses, n_frames=request.n_frames)


def build_video_path(
    source: ColmapScene,
    request: VideoPathRequest,
) -> VideoPath:
    """Build a portable fixed path bound to one canonical reconstruction."""
    K, width, height = video_camera_parameters(
        source,
        camera_source=request.camera_source,
        test_every=request.test_every,
    )
    poses = build_scene_video_poses(source, request)
    return VideoPath(
        c2w=poses.astype(np.float32),
        K=K.astype(np.float32),
        width=width,
        height=height,
        scene_fingerprint=scene_fingerprint(
            source,
            trajectory=request.trajectory,
        ),
        generation=request.metadata(),
    )
