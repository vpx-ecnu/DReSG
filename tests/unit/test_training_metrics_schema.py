from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dresg.training.output as output_module
from dresg.training.output import build_aggregate, build_training_progress

FORBIDDEN_STAGE_KEYS = {
    "mean_guidance_budget_target_l1",
    "mean_guidance_budget_max_l1",
    "mean_guidance_teacher_source_l1",
    "mean_teacher_residual_l1",
    "mean_teacher_scaled_residual_l1",
    "mean_teacher_unscaled_residual_l1",
    "teacher_residual_p",
    "teacher_residual_gamma_eff",
    "teacher_residual_log_snr",
    "teacher_residual_scale",
    "render_teacher_scaled_l1",
    "render_teacher_unscaled_l1",
    "post_setup_allocated_mb",
    "peak_extra_allocated_mb",
    "peak_reserved_mb",
    "lambda_content3d",
    "appearance_update_rule",
    "stage_dir",
    "checkpoint_elapsed_sec",
    "checkpoint_peak_allocated_mb",
}


def _sample_row() -> dict:
    return {
        "stage_index": 1,
        "prefix_length": 10,
        "timestep": 900,
        "guidance_style_loss": 1.0,
        "guidance_content_loss": 0.5,
        "teacher_l1": 0.2,
        "projection_gap_l1": 0.3,
        "fit": {
            "final_total": 0.4,
            "final_l1": 0.2,
            "final_content3d_loss": 0.05,
            "fit_elapsed_sec": 2.0,
            "fit_peak_allocated_mb": 123.0,
        },
        "effective_fit_steps": 30,
    }


def test_stage_metrics_schema_excludes_unused_diagnostics() -> None:
    row = _sample_row()

    assert FORBIDDEN_STAGE_KEYS.isdisjoint(row)
    assert set(row["fit"]) == {
        "final_total",
        "final_l1",
        "final_content3d_loss",
        "fit_elapsed_sec",
        "fit_peak_allocated_mb",
    }


def test_aggregate_rejects_empty_training_history() -> None:
    with pytest.raises(ValueError, match="empty training history"):
        build_aggregate([])


def test_aggregate_metrics_schema_excludes_unused_diagnostics() -> None:
    row = _sample_row()
    row.update(
        {
            "guidance_elapsed_since_prefix_sec": 1.0,
            "stage_elapsed_sec": 2.0,
            "train_measured_elapsed_sec": 3.0,
            "peak_allocated_mb": 456.0,
            "video_elapsed_sec": 0.0,
        }
    )

    aggregate = build_aggregate([row])

    assert FORBIDDEN_STAGE_KEYS.isdisjoint(aggregate)
    assert aggregate["mean_teacher_l1"] == 0.2
    assert aggregate["final_projection_gap_l1"] == 0.3
    assert aggregate["train_measured_elapsed_sec"] == 3.0
    assert aggregate["peak_allocated_mb"] == 456.0


def test_aggregate_rejects_missing_required_video_metrics() -> None:
    row = _sample_row()
    row.update(
        {
            "train_measured_elapsed_sec": 3.0,
            "peak_allocated_mb": 456.0,
        }
    )

    with pytest.raises(KeyError, match="video_elapsed_sec"):
        build_aggregate([row])


def test_aggregate_includes_post_color_transfer_runtime_and_peak() -> None:
    row = _sample_row()
    row.update(
        {
            "train_measured_elapsed_sec": 3.0,
            "peak_allocated_mb": 456.0,
            "post_color_transfer_elapsed_sec": 2.5,
            "post_color_transfer_peak_allocated_mb": 512.0,
            "video_peak_allocated_mb": 2048.0,
            "video_elapsed_sec": 0.0,
        }
    )

    aggregate = build_aggregate([row])

    assert aggregate["train_measured_elapsed_sec"] == 5.5
    assert aggregate["post_color_transfer_elapsed_sec"] == 2.5
    assert aggregate["peak_allocated_mb"] == 512.0


def test_training_progress_collects_final_artifact_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    saved_paths = []
    monkeypatch.setattr(
        output_module,
        "maybe_export_train_views",
        lambda **_kwargs: {"infer_fps": 2.0},
    )
    monkeypatch.setattr(
        output_module,
        "maybe_render_final_video",
        lambda **_kwargs: {"video_elapsed_sec": 3.0},
    )

    progress = build_training_progress(torch.device("cpu"), tmp_path)
    progress.stage_rows.extend([{}, {}])
    progress.completed_guidance_steps = 20
    metrics = progress._create_final_artifacts(
        scene=SimpleNamespace(save_ply=saved_paths.append),
        source=object(),
        cameras=SimpleNamespace(
            c2w=torch.empty(0),
            view_indices=(1, 3),
        ),
        artifacts=object(),
        rendering=object(),
    )

    assert saved_paths == [tmp_path / "point_cloud.ply"]
    assert metrics == {
        "infer_fps": 2.0,
        "video_elapsed_sec": 3.0,
    }


def test_video_path_presence_controls_final_rendering(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        output_module,
        "render_scene_video",
        lambda **kwargs: calls.append(kwargs),
    )
    video = SimpleNamespace(path=None)
    artifacts = SimpleNamespace(video=video)

    disabled = output_module.maybe_render_final_video(
        artifacts=artifacts,
        rendering=SimpleNamespace(render_scale=1.0),
        scene=object(),
        source=object(),
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert not calls
    assert disabled == {
        "video_elapsed_sec": 0.0,
        "video_peak_allocated_mb": 0.0,
    }

    video.path = tmp_path / "path.npz"
    enabled = output_module.maybe_render_final_video(
        artifacts=artifacts,
        rendering=SimpleNamespace(render_scale=1.0),
        scene=object(),
        source=object(),
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    assert calls[0]["video"] is video
    assert enabled["video_elapsed_sec"] >= 0.0
