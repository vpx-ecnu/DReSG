"""Strict pickle-free video-path artifact serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from dresg.data.colmap import ColmapScene
from dresg.inference.paths.trajectory import VideoPath, scene_fingerprint

_FORMAT_VERSION = 4
_ARTIFACT_FIELDS = {
    "format_version",
    "c2w",
    "K",
    "width",
    "height",
    "scene_fingerprint",
    "generation_json",
}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Video path generation_json repeats key: {name}")
        result[name] = value
    return result


def _require_npz_path(path: Path) -> Path:
    artifact_path = path.expanduser()
    if artifact_path.suffix.lower() != ".npz":
        raise ValueError("Video path artifacts must use the .npz extension")
    return artifact_path


def save_video_path(path: Path, video_path: VideoPath) -> None:
    """Atomically save one strict pickle-free camera-path artifact."""
    output = _require_npz_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generation_json = json.dumps(
        dict(video_path.generation),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}.",
        dir=output.parent,
    ) as staging_dir:
        temporary = Path(staging_dir) / output.name
        np.savez_compressed(
            temporary,
            format_version=np.asarray(_FORMAT_VERSION, dtype=np.int64),
            c2w=video_path.c2w,
            K=video_path.K,
            width=np.asarray(video_path.width, dtype=np.int64),
            height=np.asarray(video_path.height, dtype=np.int64),
            scene_fingerprint=np.asarray(video_path.scene_fingerprint),
            generation_json=np.asarray(generation_json),
        )
        temporary.replace(output)


def _scalar(payload: Any, *, name: str, dtype: np.dtype) -> Any:
    if not isinstance(payload, np.ndarray) or payload.shape != () or payload.dtype != dtype:
        raise ValueError(f"Video path {name} must be a scalar {dtype} array")
    return payload.item()


def load_video_path(path: Path) -> VideoPath:
    """Load and validate one strict pickle-free camera-path artifact."""
    artifact_path = _require_npz_path(path)
    with np.load(artifact_path, allow_pickle=False) as payload:
        actual_fields = set(payload.files)
        if actual_fields != _ARTIFACT_FIELDS:
            missing = sorted(_ARTIFACT_FIELDS - actual_fields)
            extra = sorted(actual_fields - _ARTIFACT_FIELDS)
            raise ValueError(
                f"Invalid video path fields: missing={missing}, extra={extra}"
            )
        version = _scalar(
            payload["format_version"],
            name="format_version",
            dtype=np.dtype(np.int64),
        )
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported video path format_version: {version}")
        width = _scalar(payload["width"], name="width", dtype=np.dtype(np.int64))
        height = _scalar(payload["height"], name="height", dtype=np.dtype(np.int64))
        fingerprint_array = payload["scene_fingerprint"]
        generation_array = payload["generation_json"]
        if fingerprint_array.shape != () or fingerprint_array.dtype.kind != "U":
            raise ValueError(
                "Video path scene_fingerprint must be a scalar string"
            )
        if generation_array.shape != () or generation_array.dtype.kind != "U":
            raise ValueError("Video path generation_json must be a scalar string")
        try:
            generation = json.loads(
                str(generation_array.item()),
                object_pairs_hook=_unique_json_object,
            )
        except json.JSONDecodeError as error:
            raise ValueError("Video path generation_json is invalid") from error
        if not isinstance(generation, dict):
            raise TypeError("Video path generation_json must contain an object")
        return VideoPath(
            c2w=payload["c2w"],
            K=payload["K"],
            width=int(width),
            height=int(height),
            scene_fingerprint=str(fingerprint_array.item()),
            generation=generation,
        )


def load_video_path_for_scene(
    path: Path,
    source: ColmapScene,
) -> VideoPath:
    """Load one path and verify all trajectory-dependent scene inputs."""
    video_path = load_video_path(path)
    expected = scene_fingerprint(source, trajectory=video_path.trajectory)
    if video_path.scene_fingerprint != expected:
        raise ValueError(
            "Video path scene fingerprint does not match the loaded reconstruction"
        )
    return video_path
