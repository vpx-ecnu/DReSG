from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from dresg.inference import run as run_module
from tests.config_factory import make_config


def _write_saved_config(run_dir: Path):
    config = make_config()
    config.runtime.device = "cpu"
    config.data.output_dir = Path("/original/run/location")
    config.data.scene_dir = Path("/data/scene")
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.structured(config), hydra_dir / "config.yaml")
    return config


def test_load_run_config_materializes_strict_saved_hydra_config(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    expected = _write_saved_config(run_dir)

    loaded = run_module.load_run_config(run_dir)

    assert loaded.data.output_dir == run_dir
    assert loaded.data.scene_dir == expected.data.scene_dir
    assert loaded.rendering.render_scale == expected.rendering.render_scale
    assert isinstance(loaded.data.base_ply, Path)


def test_load_run_config_rejects_unknown_saved_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_saved_config(run_dir)
    path = run_dir / ".hydra/config.yaml"
    payload = OmegaConf.load(path)
    payload.unknown = True
    OmegaConf.save(payload, path)

    with pytest.raises(ConfigKeyError, match="unknown"):
        run_module.load_run_config(run_dir)


def test_load_run_config_requires_hydra_task_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Saved Hydra config"):
        run_module.load_run_config(run_dir)


def test_render_run_train_views_uses_final_ply_and_saved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    expected = _write_saved_config(run_dir)
    source = object()
    scene = object()
    calls: dict[str, dict[str, object]] = {}

    def load_source(**kwargs):
        calls["source"] = kwargs
        return source

    def build_scene(**kwargs):
        calls["scene"] = kwargs
        return scene

    def export(**kwargs):
        calls["export"] = kwargs
        return {"render_count": 2, "infer_fps": 10.0}

    monkeypatch.setattr(run_module, "load_colmap_scene", load_source)
    monkeypatch.setattr(run_module, "build_gaussian_scene", build_scene)
    monkeypatch.setattr(run_module, "export_train_view_renders", export)

    metrics = run_module.render_run_train_views(run_dir)

    assert metrics == {"render_count": 2, "infer_fps": 10.0}
    assert calls["source"] == {
        "scene_dir": expected.data.scene_dir,
        "factor": expected.data.factor,
    }
    assert calls["scene"] == {
        "ply_path": run_dir / "point_cloud.ply",
        "device": torch.device("cpu"),
        "optimize_geometry": False,
        "optimize_quats": False,
        "max_mean_delta": 0.0,
        "max_scale_delta": 0.0,
        "max_quat_delta": 0.0,
    }
    assert calls["export"] == {
        "scene": scene,
        "source": source,
        "out_dir": run_dir,
        "render_scale": expected.rendering.render_scale,
        "device": torch.device("cpu"),
    }


def test_render_run_video_uses_saved_path_or_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    config = make_config()
    config.artifacts.video.path = tmp_path / "saved_path.npz"
    source = object()
    scene = object()
    calls = []
    monkeypatch.setattr(run_module, "load_run_config", lambda _run_dir: config)
    monkeypatch.setattr(run_module, "_load_source", lambda _config: source)
    monkeypatch.setattr(
        run_module,
        "_load_final_scene",
        lambda _run_dir, _device: scene,
    )
    monkeypatch.setattr(run_module, "load_video_path_for_scene", lambda *_args: None)
    monkeypatch.setattr(
        run_module,
        "render_scene_video",
        lambda **kwargs: calls.append(kwargs),
    )

    output = run_module.render_run_video(run_dir)
    override = tmp_path / "override_path.npz"
    run_module.render_run_video(run_dir, path=override)

    assert output == run_dir / "video.mp4"
    assert calls[0]["video"].path == config.artifacts.video.path
    assert calls[1]["video"].path == override
    assert all(call["scene"] is scene for call in calls)
    assert all(call["source"] is source for call in calls)


def test_render_run_video_requires_a_camera_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config()
    config.artifacts.video.path = None
    monkeypatch.setattr(run_module, "load_run_config", lambda _run_dir: config)
    monkeypatch.setattr(
        run_module,
        "_load_source",
        lambda _config: pytest.fail("source loaded before video path validation"),
    )

    with pytest.raises(ValueError, match="requires --path"):
        run_module.render_run_video(tmp_path / "run")
