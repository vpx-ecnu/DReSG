from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pycolmap
from numpy.typing import NDArray

from dresg.utils.images import image_size

_SUPPORTED_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
}
_PINHOLE_CAMERA_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}


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


def _relative_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required image directory does not exist: {directory}")
    return tuple(
        sorted(
            path.relative_to(directory)
            for path in directory.rglob("*")
            if path.is_file()
        )
    )


def _resolve_image_paths(
    *,
    image_names: tuple[str, ...] | list[str],
    original_dir: Path,
    pyramid_dir: Path,
) -> tuple[Path, ...]:
    """Resolve pyramid images without silently changing camera-to-image pairing."""
    image_names = tuple(image_names)
    if len(set(image_names)) != len(image_names):
        raise ValueError("COLMAP image names must be unique")
    image_stems = tuple(Path(name).with_suffix("") for name in image_names)
    if len(set(image_stems)) != len(image_stems):
        raise ValueError("COLMAP image names must have unique relative stems")

    pyramid_files = _relative_files(pyramid_dir)
    by_stem: dict[Path, Path] = {}
    for relative_path in pyramid_files:
        stem = relative_path.with_suffix("")
        if stem in by_stem:
            raise ValueError(f"Image pyramid contains duplicate relative stem: {stem}")
        by_stem[stem] = relative_path
    if all(stem in by_stem for stem in image_stems):
        return tuple(pyramid_dir / by_stem[stem] for stem in image_stems)

    original_files = _relative_files(original_dir)
    if tuple(sorted(map(Path, image_names))) != original_files:
        raise ValueError(
            "Image pyramid does not preserve COLMAP relative names, and indexed "
            "mapping requires images/ to match all registered COLMAP names"
        )
    if len(pyramid_files) != len(original_files):
        raise ValueError(
            "Indexed image pyramids require equal original and pyramid image counts: "
            f"original={len(original_files)}, pyramid={len(pyramid_files)}"
        )
    expected_stems = tuple(Path(f"image{index:03d}") for index in range(len(pyramid_files)))
    actual_stems = tuple(path.with_suffix("") for path in pyramid_files)
    if actual_stems != expected_stems:
        raise ValueError(
            "Image pyramid must preserve COLMAP relative names or use "
            "image000, image001, ... indexed names: "
            f"actual={list(actual_stems[:5])}"
        )
    by_original_name = dict(zip(original_files, pyramid_files, strict=True))
    return tuple(pyramid_dir / by_original_name[Path(name)] for name in image_names)


def _load_reconstruction(colmap_dir: Path) -> pycolmap.Reconstruction:
    reconstruction_type = getattr(pycolmap, "Reconstruction", None)
    if reconstruction_type is None:
        raise ImportError(
            "DReSG requires the official pycolmap package (>=3.12). "
            "Uninstall the unrelated pycolmap 0.0.1 package and reinstall requirements.txt."
        )
    return reconstruction_type(colmap_dir)


def _camera_model_name(camera: pycolmap.Camera) -> str:
    model = camera.model
    return str(getattr(model, "name", model)).replace("CameraModelId.", "")


def _readonly_float64(array: NDArray[np.floating]) -> NDArray[np.float64]:
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _scaled_calibration_matrix(
    camera: pycolmap.Camera,
    *,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    scaled = pycolmap.Camera(camera.todict())
    scaled.rescale(width, height)
    calibration = np.asarray(scaled.calibration_matrix(), dtype=np.float64)
    return _readonly_float64(calibration)


@dataclass(frozen=True)
class ColmapScene:
    """Validated camera metadata and resolved image paths for one COLMAP scene."""

    scene_dir: Path
    factor: int
    image_names: tuple[str, ...]
    image_paths: tuple[Path, ...]
    camtoworlds: NDArray[np.float64]
    camera_ids: tuple[int, ...]
    intrinsics_by_camera: Mapping[int, NDArray[np.float64]]
    image_sizes_by_camera: Mapping[int, tuple[int, int]]

    def __post_init__(self) -> None:
        count = len(self.image_names)
        if count == 0:
            raise ValueError("COLMAP scene must contain at least one registered image")
        if not (
            count
            == len(self.image_paths)
            == len(self.camera_ids)
            == self.camtoworlds.shape[0]
        ):
            raise ValueError("COLMAP image, path, pose, and camera-ID counts must match")
        if len(set(self.image_names)) != count:
            raise ValueError("COLMAP image names must be unique")
        if self.camtoworlds.shape != (count, 4, 4):
            raise ValueError("camtoworlds must have shape [N, 4, 4]")
        if not np.isfinite(self.camtoworlds).all():
            raise ValueError("camtoworlds must contain only finite values")
        _require_positive_integer(self.factor, name="factor")
        for camera_id in self.camera_ids:
            _require_nonnegative_integer(camera_id, name="camera_ids entries")
        missing_paths = [path for path in self.image_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Resolved scene images do not exist: {missing_paths[:5]}")

        intrinsics: dict[int, NDArray[np.float64]] = {}
        sizes: dict[int, tuple[int, int]] = {}
        for camera_id in set(self.camera_ids):
            if camera_id not in self.intrinsics_by_camera:
                raise ValueError(f"Missing intrinsics for COLMAP camera {camera_id}")
            if camera_id not in self.image_sizes_by_camera:
                raise ValueError(f"Missing image size for COLMAP camera {camera_id}")
            K = _readonly_float64(self.intrinsics_by_camera[camera_id])
            if K.shape != (3, 3) or not np.isfinite(K).all():
                raise ValueError(f"Invalid intrinsics for COLMAP camera {camera_id}")
            if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
                raise ValueError(f"Focal lengths must be positive for camera {camera_id}")
            width, height = self.image_sizes_by_camera[camera_id]
            _require_positive_integer(width, name=f"Camera {camera_id} width")
            _require_positive_integer(height, name=f"Camera {camera_id} height")
            intrinsics[camera_id] = K
            sizes[camera_id] = (width, height)

        object.__setattr__(self, "camtoworlds", _readonly_float64(self.camtoworlds))
        object.__setattr__(self, "intrinsics_by_camera", MappingProxyType(intrinsics))
        object.__setattr__(self, "image_sizes_by_camera", MappingProxyType(sizes))

    def __len__(self) -> int:
        return len(self.image_names)

    def validate_view_index(self, view_index: int) -> None:
        index = _require_nonnegative_integer(view_index, name="Camera view index")
        if index >= len(self):
            raise ValueError(
                f"Camera view index {index} is outside the valid range 0..{len(self) - 1}"
            )


def load_colmap_scene(
    *,
    scene_dir: Path,
    factor: int,
) -> ColmapScene:
    """Load camera metadata and resolve canonical image paths for one scene."""
    if not isinstance(scene_dir, Path):
        raise TypeError("scene_dir must be a pathlib.Path")
    _require_positive_integer(factor, name="factor")
    colmap_dir = scene_dir / "sparse" / "0"
    if not colmap_dir.is_dir():
        raise FileNotFoundError(f"Required COLMAP directory does not exist: {colmap_dir}")

    reconstruction = _load_reconstruction(colmap_dir)
    images = {int(image_id): image for image_id, image in reconstruction.images.items()}
    image_ids = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())
    if not image_ids:
        raise ValueError("No registered images found in COLMAP reconstruction")
    missing_image_ids = [image_id for image_id in image_ids if image_id not in images]
    if missing_image_ids:
        raise ValueError(f"Registered COLMAP image IDs are missing: {missing_image_ids}")

    records: list[tuple[str, int, NDArray[np.float64]]] = []
    for image_id in image_ids:
        image = images[image_id]
        camera_id = int(image.camera_id)
        if camera_id not in reconstruction.cameras:
            raise ValueError(f"COLMAP image {image_id} references missing camera {camera_id}")
        cam_from_world = image.cam_from_world()
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :4] = np.asarray(cam_from_world.inverse().matrix(), dtype=np.float64)
        records.append((str(image.name), camera_id, c2w))
    records.sort(key=lambda record: record[0])
    image_names = tuple(record[0] for record in records)
    if len(set(image_names)) != len(image_names):
        raise ValueError("COLMAP image names must be unique")
    camera_ids = tuple(record[1] for record in records)
    camtoworlds = np.stack([record[2] for record in records], axis=0)

    original_dir = scene_dir / "images"
    pyramid_dir = original_dir if factor == 1 else scene_dir / f"images_{factor}"
    if not original_dir.is_dir():
        raise FileNotFoundError(f"Required image directory does not exist: {original_dir}")
    image_paths = _resolve_image_paths(
        image_names=image_names,
        original_dir=original_dir,
        pyramid_dir=pyramid_dir,
    )

    actual_sizes_by_camera: dict[int, set[tuple[int, int]]] = {}
    for camera_id, image_path in zip(camera_ids, image_paths, strict=True):
        actual_sizes_by_camera.setdefault(camera_id, set()).add(image_size(image_path))

    intrinsics_by_camera: dict[int, NDArray[np.float64]] = {}
    image_sizes_by_camera: dict[int, tuple[int, int]] = {}
    has_distortion = False
    for camera_id in sorted(set(camera_ids)):
        camera = reconstruction.cameras[camera_id]
        camera_model = _camera_model_name(camera)
        if camera_model not in _SUPPORTED_CAMERA_MODELS:
            raise ValueError(f"Unsupported COLMAP camera model: {camera_model}")
        has_distortion = has_distortion or camera_model not in _PINHOLE_CAMERA_MODELS
        actual_sizes = actual_sizes_by_camera[camera_id]
        if len(actual_sizes) != 1:
            raise ValueError(
                "Images assigned to one COLMAP camera must share a resolution; "
                f"camera {camera_id} has {sorted(actual_sizes)}"
            )
        actual_width, actual_height = next(iter(actual_sizes))
        intrinsics_by_camera[camera_id] = _scaled_calibration_matrix(
            camera,
            width=actual_width,
            height=actual_height,
        )
        image_sizes_by_camera[camera_id] = (actual_width, actual_height)

    if has_distortion:
        warnings.warn(
            "COLMAP cameras include distortion. DReSG uses the same pinhole "
            "approximation as the supplied Gaussian PLY; the image set "
            "must match that reconstruction.",
            stacklevel=2,
        )

    return ColmapScene(
        scene_dir=scene_dir,
        factor=factor,
        image_names=image_names,
        image_paths=image_paths,
        camtoworlds=camtoworlds,
        camera_ids=camera_ids,
        intrinsics_by_camera=intrinsics_by_camera,
        image_sizes_by_camera=image_sizes_by_camera,
    )
