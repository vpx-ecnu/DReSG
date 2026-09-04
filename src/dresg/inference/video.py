"""Render scene videos from validated fixed camera-path artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tqdm import tqdm

from dresg.config import VideoConfig
from dresg.data.cameras import scaled_intrinsics
from dresg.data.colmap import ColmapScene
from dresg.inference.paths import load_video_path_for_scene
from dresg.models.gs import GaussianScene
from dresg.utils.images import chw_to_hwc_u8
from dresg.utils.video import save_video


def _render_frames(
    *,
    scene: GaussianScene,
    poses: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    for start in tqdm(
        range(0, len(poses), batch_size),
        desc="Rendering scene video",
    ):
        c2w = poses[start : start + batch_size]
        K_batch = K.unsqueeze(0).expand(c2w.shape[0], -1, -1)
        frames = scene.render_batch(
            c2w=c2w,
            K=K_batch,
            width=width,
            height=height,
        )
        for frame in frames.detach().cpu():
            yield chw_to_hwc_u8(frame.permute(2, 0, 1))


@torch.no_grad()
def render_scene_video(
    *,
    scene: GaussianScene,
    source: ColmapScene,
    output_path: Path,
    video: VideoConfig,
    render_scale: float,
    device: torch.device,
) -> None:
    path = load_video_path_for_scene(cast(Path, video.path), source)
    poses = torch.tensor(path.c2w, device=device)
    K = torch.tensor(path.K, device=device)
    K, width, height = scaled_intrinsics(
        K,
        path.width,
        path.height,
        render_scale,
    )
    save_video(
        output_path,
        _render_frames(
            scene=scene,
            poses=poses,
            K=K,
            width=width,
            height=height,
            batch_size=video.batch_size,
        ),
        fps=video.fps,
    )
