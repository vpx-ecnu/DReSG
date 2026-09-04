"""Fixed-scene view and video inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dresg.inference.run import render_run_train_views, render_run_video
    from dresg.inference.video import render_scene_video
    from dresg.inference.views import export_train_view_renders

__all__ = (
    "export_train_view_renders",
    "render_run_train_views",
    "render_run_video",
    "render_scene_video",
)


def __getattr__(name: str):
    if name == "export_train_view_renders":
        from dresg.inference.views import export_train_view_renders

        return export_train_view_renders
    if name in {"render_run_train_views", "render_run_video"}:
        from dresg.inference.run import render_run_train_views, render_run_video

        return {
            "render_run_train_views": render_run_train_views,
            "render_run_video": render_run_video,
        }[name]
    if name == "render_scene_video":
        from dresg.inference.video import render_scene_video

        return render_scene_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
