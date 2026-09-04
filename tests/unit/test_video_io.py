from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dresg.utils.video as video_io


class _Writer:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True
        self.path.write_bytes(b"video")


def _patch_writer(monkeypatch) -> list[_Writer]:
    writers: list[_Writer] = []

    def factory(path: str, **_kwargs) -> _Writer:
        writer = _Writer(path)
        writers.append(writer)
        return writer

    monkeypatch.setattr(video_io.imageio, "get_writer", factory)
    return writers


def test_video_io_pads_frames_and_commits_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writers = _patch_writer(monkeypatch)
    output = tmp_path / "result.mp4"
    frames = [np.full((3, 5, 3), index, dtype=np.uint8) for index in range(2)]

    video_io.save_video(output, frames, fps=24)

    assert output.read_bytes() == b"video"
    assert writers[0].closed
    assert [frame.shape for frame in writers[0].frames] == [(4, 6, 3)] * 2
    assert not list(tmp_path.glob(".*.mp4"))


def test_video_io_preserves_existing_output_when_frames_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writers = _patch_writer(monkeypatch)
    output = tmp_path / "result.mp4"
    output.write_bytes(b"existing")

    def frames():
        yield np.zeros((4, 6, 3), dtype=np.uint8)
        raise RuntimeError("frame generation failed")

    with pytest.raises(RuntimeError, match="frame generation failed"):
        video_io.save_video(output, frames(), fps=24)

    assert output.read_bytes() == b"existing"
    assert writers[0].closed
    assert not list(tmp_path.glob(".*.mp4"))


def test_video_io_rejects_empty_frame_stream(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writers = _patch_writer(monkeypatch)

    with pytest.raises(ValueError, match="must not be empty"):
        video_io.save_video(tmp_path / "result.mp4", [], fps=24)

    assert writers[0].closed
    assert not list(tmp_path.glob(".*.mp4"))


def test_video_io_rejects_inconsistent_frame_shapes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_writer(monkeypatch)
    frames = [
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.zeros((6, 4, 3), dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="share one shape"):
        video_io.save_video(tmp_path / "result.mp4", frames, fps=24)
