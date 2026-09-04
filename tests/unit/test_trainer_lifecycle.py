from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import dresg.training.output as output_module
import dresg.training.trainer as trainer_module
from dresg.training.optimization.guidance import GuidanceMetrics
from dresg.training.output import build_training_progress
from dresg.training.trainer import DReSGTrainer
from dresg.utils.json_io import load_json
from tests.config_factory import make_config


def _runtime(
    *,
    output_dir: Path,
    active_prefixes: tuple[int, ...] = (1,),
    post_color_enabled: bool = True,
):
    config = SimpleNamespace(
        artifacts=SimpleNamespace(),
        rendering=object(),
        debug=SimpleNamespace(collect_stage_diagnostics=False),
        schedule=SimpleNamespace(
            active_prefixes=active_prefixes,
            fit_steps=4,
        ),
        guidance=SimpleNamespace(optimization=object()),
        image_loss=object(),
        appearance_update=SimpleNamespace(rule="pcgrad"),
        color_transfer=SimpleNamespace(
            post_enabled=post_color_enabled,
            post_fit_steps=9,
        ),
        data=SimpleNamespace(
            output_dir=output_dir,
            style_image=Path("/tmp/style.png"),
        ),
    )
    source = object()
    cameras = SimpleNamespace(
        c2w=torch.empty(0, device="cpu"),
        view_indices=(),
    )
    scene = SimpleNamespace(parameter_stats=lambda: {})
    scene_optimizer = object()
    base_renders = {}
    progress = build_training_progress(torch.device("cpu"), output_dir)
    return (
        config,
        source,
        cameras,
        scene,
        scene_optimizer,
        base_renders,
        progress,
    )


def _trainer(runtime, guidance) -> DReSGTrainer:
    (
        config,
        source,
        cameras,
        scene,
        scene_optimizer,
        base_renders,
        progress,
    ) = runtime
    trainer = DReSGTrainer.__new__(DReSGTrainer)
    trainer.config = config
    trainer.source = source
    trainer.cameras = cameras
    trainer.scene = scene
    trainer.scene_optimizer = scene_optimizer
    trainer.base_renders = base_renders
    trainer.progress = progress
    trainer.guidance = guidance
    trainer.content_loss = object()
    return trainer


def test_trainer_initializes_owned_runtime_from_config(
    monkeypatch,
    tmp_path,
) -> None:
    config = make_config()
    config.image_loss.lambda_content3d = 0.0
    config.data.output_dir = tmp_path
    config.rendering.teacher_width = 64
    config.rendering.teacher_height = 64
    config.artifacts.save_style_image = False
    base_renders = {4: torch.zeros(3, 2, 2)}
    source_images = {4: torch.ones(3, 2, 2)}
    source = object()
    cameras = SimpleNamespace(
        c2w=torch.empty(0, device="cpu"),
        view_indices=(4,),
    )
    scene = SimpleNamespace(
        render_current_images=lambda _cameras: base_renders,
    )
    scene_optimizer = object()
    guidance = SimpleNamespace(view_ids=(4,))
    calls: list[dict[str, object]] = []
    seed_calls: list[int] = []

    monkeypatch.setattr(trainer_module, "validate_run_config", lambda _config: None)
    monkeypatch.setattr(
        trainer_module,
        "seed_random_generators",
        seed_calls.append,
    )
    monkeypatch.setattr(
        trainer_module,
        "load_colmap_scene",
        lambda **_kwargs: source,
    )
    monkeypatch.setattr(
        trainer_module,
        "build_scaled_cameras",
        lambda **_kwargs: cameras,
    )
    monkeypatch.setattr(
        trainer_module,
        "build_gaussian_scene",
        lambda **_kwargs: scene,
    )
    monkeypatch.setattr(
        trainer_module,
        "GaussianOptimizer",
        lambda *_args: scene_optimizer,
    )
    monkeypatch.setattr(
        trainer_module,
        "load_source_view_images",
        lambda *_args: source_images,
    )
    monkeypatch.setattr(
        trainer_module,
        "load_rgb_image",
        lambda *_args, **_kwargs: torch.ones(3, 2, 2),
    )

    def load_guidance(**kwargs):
        calls.append(kwargs)
        return guidance

    monkeypatch.setattr(
        trainer_module,
        "build_diffusion_guidance",
        load_guidance,
    )

    trainer = DReSGTrainer(config)

    assert trainer.config is config
    assert trainer.source is source
    assert trainer.cameras is cameras
    assert trainer.scene is scene
    assert trainer.scene_optimizer is scene_optimizer
    assert trainer.base_renders is base_renders
    assert seed_calls == [config.runtime.seed]
    assert trainer.guidance is guidance
    assert calls[0]["base_renders_by_view"] is base_renders
    assert calls[0]["source_images_by_view"] is source_images
    assert calls[0]["style_image"].shape == (3, 2, 2)


def test_guidance_feedback_precedes_post_color_and_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[str] = []
    trainer = _trainer(_runtime(output_dir=tmp_path), SimpleNamespace(timesteps=()))

    def run_guidance() -> None:
        events.append("guidance-feedback")
        trainer.progress.stage_rows.append({})

    trainer._run_guidance_stages = run_guidance
    trainer._run_post_color_stage = lambda: events.append("post")
    monkeypatch.setattr(
        trainer.progress,
        "_create_final_artifacts",
        lambda **_kwargs: events.append("artifacts") or {},
    )
    monkeypatch.setattr(output_module, "build_aggregate", lambda _rows: {})

    trainer.run()

    assert events == ["guidance-feedback", "post", "artifacts"]


def test_trainer_centrally_schedules_guidance_and_post_color_stages(
    tmp_path,
    monkeypatch,
) -> None:
    guidance = SimpleNamespace(timesteps=(torch.tensor(900), torch.tensor(800), torch.tensor(700)))
    trainer = _trainer(
        _runtime(output_dir=tmp_path, active_prefixes=(1, 3)),
        guidance,
    )
    events: list[tuple[str, int]] = []
    stage_requests: list[dict[str, int]] = []
    post_requests: list[dict[str, int]] = []
    progress_options: dict[str, object] = {}
    progress_updates: list[int] = []
    progress_metrics: list[dict[str, object]] = []
    progress_postfixes: list[str] = []

    class FakeProgressBar:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def update(self) -> None:
            progress_updates.append(1)

        def set_postfix(self, **metrics) -> None:
            progress_metrics.append(metrics)

        def set_postfix_str(self, value: str = "") -> None:
            progress_postfixes.append(value)

    def fake_tqdm(**kwargs):
        progress_options.update(kwargs)
        return FakeProgressBar()

    class FakeGuidanceStage:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, timestep):
            events.append(("guidance", int(timestep.item())))
            return [GuidanceMetrics(1.0, 0.5, 1.5, view_count=1)]

    class FakeFeedbackStage:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, **kwargs):
            events.append(("gaussian", kwargs["prefix"]))
            stage_requests.append(kwargs)
            return {
                "prefix_length": kwargs["prefix"],
                "timestep": int(guidance.timesteps[kwargs["prefix"] - 1].item()),
                "teacher_l1": 0.2,
                "projection_gap_l1": 0.3,
                "post_color_transfer_enabled": 0,
            }

    class FakeColorStage:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, **kwargs):
            post_requests.append(kwargs)
            return {"post_color_transfer_enabled": 1}

    monkeypatch.setattr(trainer_module, "tqdm", fake_tqdm)
    monkeypatch.setattr(trainer_module, "GuidanceStage", FakeGuidanceStage)
    monkeypatch.setattr(trainer_module, "FeedbackStage", FakeFeedbackStage)
    monkeypatch.setattr(trainer_module, "ColorStage", FakeColorStage)
    monkeypatch.setattr(
        trainer.progress,
        "_create_final_artifacts",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        output_module,
        "build_aggregate",
        lambda rows: {"num_stages": len(rows)},
    )

    trainer.run()

    assert progress_options == {
        "total": 3,
        "desc": "DReSG guidance",
        "unit": "step",
        "dynamic_ncols": True,
        "disable": None,
    }
    assert progress_updates == [1, 1, 1]
    assert progress_metrics == [
        {"prefix": 1, "style": "1.0000", "content": "0.5000"},
        {"prefix": 1, "teacher": "0.2000", "projection": "0.3000"},
        {"prefix": 3, "style": "1.0000", "content": "0.5000"},
        {"prefix": 3, "style": "1.0000", "content": "0.5000"},
        {"prefix": 3, "teacher": "0.2000", "projection": "0.3000"},
    ]
    assert progress_postfixes == [
        "feedback prefix=1",
        "feedback prefix=3",
    ]
    assert stage_requests == [
        {
            "prefix": 1,
            "fit_steps": 4,
        },
        {
            "prefix": 3,
            "fit_steps": 4,
        },
    ]
    assert events == [
        ("guidance", 900),
        ("gaussian", 1),
        ("guidance", 800),
        ("guidance", 700),
        ("gaussian", 3),
    ]
    assert post_requests == [{"fit_steps": 9}]
    assert trainer.progress.stage_rows == [
        {
            "stage_index": 1,
            "prefix_length": 1,
            "timestep": 900,
            "guidance_style_loss": 1.0,
            "guidance_content_loss": 0.5,
            "teacher_l1": 0.2,
            "projection_gap_l1": 0.3,
            "post_color_transfer_enabled": 0,
        },
        {
            "stage_index": 2,
            "prefix_length": 3,
            "timestep": 700,
            "guidance_style_loss": 1.0,
            "guidance_content_loss": 0.5,
            "teacher_l1": 0.2,
            "projection_gap_l1": 0.3,
            "post_color_transfer_enabled": 1,
        },
    ]


def test_trainer_skips_disabled_post_color_stage(tmp_path, monkeypatch) -> None:
    trainer = _trainer(
        _runtime(output_dir=tmp_path, post_color_enabled=False),
        SimpleNamespace(timesteps=()),
    )
    trainer._run_guidance_stages = lambda: trainer.progress.stage_rows.append({})
    trainer._run_post_color_stage = lambda: pytest.fail("disabled post-color stage must not execute")
    monkeypatch.setattr(
        trainer.progress,
        "_create_final_artifacts",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(output_module, "build_aggregate", lambda _rows: {})

    trainer.run()


def test_guidance_failure_skips_later_phases(tmp_path, monkeypatch) -> None:
    events: list[str] = []
    guidance = SimpleNamespace(timesteps=(torch.tensor(1),))
    trainer = _trainer(_runtime(output_dir=tmp_path), guidance)

    class FailingGuidanceStage:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, _timestep) -> None:
            raise RuntimeError("guidance failed")

    monkeypatch.setattr(
        trainer_module,
        "GuidanceStage",
        FailingGuidanceStage,
    )
    trainer._run_post_color_stage = lambda: events.append("post")
    monkeypatch.setattr(
        trainer.progress,
        "_create_final_artifacts",
        lambda **_kwargs: events.append("artifacts"),
    )

    with pytest.raises(RuntimeError, match="guidance failed"):
        trainer.run()

    assert events == []
    summary = load_json(tmp_path / "summary.json")
    assert summary["status"] == "running"
    assert summary["completed_guidance_steps"] == 0
    assert not (tmp_path / "aggregate_metrics.json").exists()
