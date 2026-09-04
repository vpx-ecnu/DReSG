"""Strict atomic MP4 encoding for RGB frame streams."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _validate_frame(
    frame: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    if not isinstance(frame, np.ndarray):
        raise TypeError("Video frames must be NumPy arrays")
    if frame.dtype != np.uint8:
        raise TypeError("Video frames must use uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Video frames must have shape [H, W, 3]")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("Video frame dimensions must be positive")
    if expected_shape is not None and frame.shape != expected_shape:
        raise ValueError("All video frames must share one shape")
    return frame.shape


def _make_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_video(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int,
) -> None:
    """Encode one non-empty RGB frame stream and atomically replace its MP4."""
    if not isinstance(path, Path):
        raise TypeError("Video output path must be a Path")
    if path.suffix.lower() != ".mp4":
        raise ValueError("Video output path must use the .mp4 extension")
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise TypeError("Video fps must be an integer")
    if fps < 1:
        raise ValueError("Video fps must be positive")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.stem}.",
        dir=path.parent,
    ) as staging_dir:
        temporary = Path(staging_dir) / path.name
        with closing(
            imageio.get_writer(
                str(temporary),
                fps=fps,
                macro_block_size=1,
            )
        ) as writer:
            frame_count = 0
            expected_shape: tuple[int, int, int] | None = None
            for frame in frames:
                expected_shape = _validate_frame(
                    frame,
                    expected_shape=expected_shape,
                )
                writer.append_data(_make_even(frame))
                frame_count += 1
        if frame_count == 0:
            raise ValueError("Video frame stream must not be empty")
        temporary.replace(path)
