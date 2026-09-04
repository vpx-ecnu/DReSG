from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import dresg.inference.video as video_module
from dresg.inference.paths import VideoPath
from tests.config_factory import make_config


def _video_path(frame_count: int = 3) -> VideoPath:
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    return VideoPath(
        c2w=c2w,
        K=np.array(
            [[10.0, 0.0, 2.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        width=5,
        height=3,
        scene_fingerprint="a" * 64,
        generation={
            "trajectory": "interpolated",
            "camera_source": "all",
            "n_frames": frame_count,
        },
    )


class _Scene:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.K_batch_strides: list[int] = []

    def render_batch(
        self,
        *,
        c2w: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        self.batch_sizes.append(c2w.shape[0])
        self.K_batch_strides.append(K.stride(0))
        return torch.full(
            (c2w.shape[0], height, width, 3),
            0.5,
            device=c2w.device,
        )


def test_video_inference_batches_fixed_path_and_delegates_encoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _video_path()
    monkeypatch.setattr(
        video_module,
        "load_video_path_for_scene",
        lambda *_args: path,
    )
    encoded: dict[str, object] = {}

    def save(
        output: Path,
        frames,
        *,
        fps: int,
    ) -> None:
        encoded["output"] = output
        encoded["frames"] = list(frames)
        encoded["fps"] = fps

    monkeypatch.setattr(video_module, "save_video", save)
    config = make_config().artifacts.video
    config.batch_size = 2
    output = tmp_path / "result.mp4"
    scene = _Scene()

    video_module.render_scene_video(
        scene=scene,
        source=SimpleNamespace(),
        output_path=output,
        video=config,
        render_scale=1.0,
        device=torch.device("cpu"),
    )

    assert scene.batch_sizes == [2, 1]
    assert scene.K_batch_strides[0] == 0
    assert encoded["output"] == output
    assert encoded["fps"] == config.fps
    frames = encoded["frames"]
    assert isinstance(frames, list)
    assert [frame.shape for frame in frames] == [(3, 5, 3)] * 3
    assert all(frame.dtype == np.uint8 for frame in frames)
