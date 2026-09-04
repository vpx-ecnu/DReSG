from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import torch

from dresg.data.cameras import build_scaled_cameras
from dresg.data.colmap import ColmapScene
from dresg.models.gs import GaussianScene
from dresg.utils.images import save_rgb
from dresg.utils.json_io import save_json
from dresg.utils.results import (
    EXPORT_METRICS_FILENAME,
    RENDERS_DIRNAME,
    VIEW_MANIFEST_FILENAME,
    ViewManifestEntry,
    export_metrics_path,
    render_filename,
    renders_dir,
    write_view_manifest,
)
from dresg.utils.runtime_metrics import memory_snapshot, reset_peak_memory, synchronize_device

_EXPORT_NAMES = (
    RENDERS_DIRNAME,
    EXPORT_METRICS_FILENAME,
    VIEW_MANIFEST_FILENAME,
)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _commit_export(staging: Path, destination: Path) -> None:
    backup = Path(
        tempfile.mkdtemp(
            prefix=".views-backup.",
            dir=destination,
        )
    )
    moved: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for name in _EXPORT_NAMES:
            target = destination / name
            if target.exists() or target.is_symlink():
                saved = backup / name
                target.replace(saved)
                moved.append((target, saved))
        for name in _EXPORT_NAMES:
            target = destination / name
            (staging / name).replace(target)
            installed.append(target)
    except BaseException:
        for target in reversed(installed):
            _remove_path(target)
        for target, saved in reversed(moved):
            saved.replace(target)
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)


@torch.no_grad()
def export_train_view_renders(
    *,
    scene: GaussianScene,
    source: ColmapScene,
    out_dir: Path,
    render_scale: float,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Render every input camera and transactionally replace its export bundle."""
    view_ids = list(range(len(source)))
    cameras = build_scaled_cameras(
        source=source,
        view_ids=view_ids,
        device=device,
        render_scale=render_scale,
        label="train export",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".views.", dir=out_dir) as staging_dir:
        staging = Path(staging_dir)
        output_renders_dir = renders_dir(staging)
        output_renders_dir.mkdir(parents=True, exist_ok=True)
        manifest_entries: list[ViewManifestEntry] = []

        reset_peak_memory(device)
        render_elapsed_sec = 0.0
        for camera in cameras:
            view_index = camera.view_index
            synchronize_device(device)
            started = time.perf_counter()
            render = scene.render(camera)
            synchronize_device(device)
            render_elapsed_sec += time.perf_counter() - started

            render_file = render_filename(view_index)
            save_rgb(output_renders_dir / render_file, render)
            manifest_entries.append(
                ViewManifestEntry(
                    render_file=render_file,
                    view_index=view_index,
                )
            )

        manifest_path = write_view_manifest(staging, manifest_entries)
        snapshot = memory_snapshot(device)
        render_count = len(manifest_entries)
        metrics: dict[str, float | int | str] = {
            "render_count": render_count,
            "render_width": cameras.width,
            "render_height": cameras.height,
            "pure_infer_elapsed_sec": render_elapsed_sec,
            "infer_fps": render_count / render_elapsed_sec if render_elapsed_sec > 0.0 else 0.0,
            "infer_peak_mem_mb": snapshot.peak_allocated_mb,
            "view_manifest": manifest_path.name,
        }
        save_json(export_metrics_path(staging), metrics)
        _commit_export(staging, out_dir)
        return metrics
