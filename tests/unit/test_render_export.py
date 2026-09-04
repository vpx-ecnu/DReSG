from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import dresg.inference.views as views_module
from dresg.data.cameras import Cameras
from dresg.inference import export_train_view_renders


class FakeScene:
    def __init__(self, *, fail_view: int | None = None) -> None:
        self.fail_view = fail_view

    def render(self, camera: object) -> torch.Tensor:
        if camera.view_index == self.fail_view:
            raise RuntimeError("render failed")
        value = float(camera.view_index) / 10.0
        return torch.full((3, 4, 6), value)


class FakeSource:
    def __len__(self) -> int:
        return 2


def _cameras() -> Cameras:
    return Cameras(
        view_indices=(0, 1),
        c2w=torch.eye(4).repeat(2, 1, 1),
        K=torch.eye(3).repeat(2, 1, 1),
        width=6,
        height=4,
    )


def _patch_cameras(monkeypatch) -> None:
    monkeypatch.setattr(
        "dresg.inference.views.build_scaled_cameras",
        lambda **_kwargs: _cameras(),
    )


def test_export_train_views_replaces_complete_bundle_and_removes_stale_renders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_cameras(monkeypatch)
    run_dir = tmp_path / "run"
    stale = run_dir / "renders" / "999999.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    metrics = export_train_view_renders(
        scene=FakeScene(),
        source=FakeSource(),
        out_dir=run_dir,
        render_scale=1.0,
        device=torch.device("cpu"),
    )

    assert sorted(path.name for path in (run_dir / "renders").glob("*.png")) == [
        "000000.png",
        "000001.png",
    ]
    assert not stale.exists()
    with (run_dir / "view_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"render_file": "000000.png", "view_index": "0"},
        {"render_file": "000001.png", "view_index": "1"},
    ]

    saved_metrics = json.loads((run_dir / "export_metrics.json").read_text())
    assert metrics["render_count"] == 2
    assert saved_metrics["render_width"] == 6
    assert saved_metrics["render_height"] == 4
    assert saved_metrics["infer_fps"] > 0.0
    assert saved_metrics["view_manifest"] == "view_manifest.csv"
    assert not list(run_dir.glob(".views.*"))



def test_view_export_commit_restores_previous_bundle_on_install_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "run"
    staging = tmp_path / "staging"
    for root, label in ((destination, "old"), (staging, "new")):
        renders = root / "renders"
        renders.mkdir(parents=True)
        (renders / "image.png").write_text(label)
        (root / "export_metrics.json").write_text(f"{label} metrics")
        (root / "view_manifest.csv").write_text(f"{label} manifest")

    original_replace = Path.replace

    def replace_path(path: Path, target: Path) -> Path:
        if path == staging / "export_metrics.json":
            raise OSError("install failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_path)

    with pytest.raises(OSError, match="install failed"):
        views_module._commit_export(staging, destination)

    assert (destination / "renders" / "image.png").read_text() == "old"
    assert (destination / "export_metrics.json").read_text() == "old metrics"
    assert (destination / "view_manifest.csv").read_text() == "old manifest"
    assert not list(destination.glob(".views-backup.*"))


def test_view_export_keeps_backup_when_rollback_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "run"
    staging = tmp_path / "staging"
    for root, label in ((destination, "old"), (staging, "new")):
        renders = root / "renders"
        renders.mkdir(parents=True)
        (renders / "image.png").write_text(label)
        (root / "export_metrics.json").write_text(f"{label} metrics")
        (root / "view_manifest.csv").write_text(f"{label} manifest")

    original_replace = Path.replace

    def replace_path(path: Path, target: Path) -> Path:
        if path == staging / "export_metrics.json":
            raise OSError("install failed")
        if path.parent.name.startswith(".views-backup") and path.name == "renders":
            raise OSError("restore failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_path)

    with pytest.raises(OSError, match="restore failed"):
        views_module._commit_export(staging, destination)

    backups = list(destination.glob(".views-backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "renders" / "image.png").read_text() == "old"


def test_export_train_views_preserves_existing_bundle_on_render_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_cameras(monkeypatch)
    run_dir = tmp_path / "run"
    renders = run_dir / "renders"
    renders.mkdir(parents=True)
    old_render = renders / "old.png"
    old_manifest = run_dir / "view_manifest.csv"
    old_metrics = run_dir / "export_metrics.json"
    old_render.write_bytes(b"old render")
    old_manifest.write_bytes(b"old manifest")
    old_metrics.write_bytes(b"old metrics")

    with pytest.raises(RuntimeError, match="render failed"):
        export_train_view_renders(
            scene=FakeScene(fail_view=1),
            source=FakeSource(),
            out_dir=run_dir,
            render_scale=1.0,
            device=torch.device("cpu"),
        )

    assert [path.name for path in renders.iterdir()] == ["old.png"]
    assert old_render.read_bytes() == b"old render"
    assert old_manifest.read_bytes() == b"old manifest"
    assert old_metrics.read_bytes() == b"old metrics"
    assert not list(run_dir.glob(".views.*"))
