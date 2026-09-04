"""Camera-trajectory construction and persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dresg.inference.paths.codec import (
        load_video_path,
        load_video_path_for_scene,
        save_video_path,
    )
    from dresg.inference.paths.trajectory import (
        VideoPath,
        VideoPathRequest,
        build_video_path,
    )

__all__ = (
    "VideoPath",
    "VideoPathRequest",
    "build_video_path",
    "load_video_path",
    "load_video_path_for_scene",
    "save_video_path",
)


def __getattr__(name: str):
    if name in {"VideoPath", "VideoPathRequest", "build_video_path"}:
        from dresg.inference.paths.trajectory import (
            VideoPath,
            VideoPathRequest,
            build_video_path,
        )

        return {
            "VideoPath": VideoPath,
            "VideoPathRequest": VideoPathRequest,
            "build_video_path": build_video_path,
        }[name]
    if name in {
        "load_video_path",
        "load_video_path_for_scene",
        "save_video_path",
    }:
        from dresg.inference.paths.codec import (
            load_video_path,
            load_video_path_for_scene,
            save_video_path,
        )

        return {
            "load_video_path": load_video_path,
            "load_video_path_for_scene": load_video_path_for_scene,
            "save_video_path": save_video_path,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
