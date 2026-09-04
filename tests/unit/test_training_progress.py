from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dresg.training.output as output_module
from dresg.training.optimization.guidance import GuidanceMetrics
from dresg.training.output import build_training_progress
from dresg.utils.json_io import load_json


def test_training_progress_persists_each_metric_update(
    tmp_path,
    monkeypatch,
) -> None:
    written_paths = []
    save_json = output_module.save_json

    def track_write(path, payload) -> None:
        written_paths.append(path)
        save_json(path, payload)

    monkeypatch.setattr(output_module, "save_json", track_write)
    progress = build_training_progress(torch.device("cpu"), tmp_path)
    first_metrics = [
        GuidanceMetrics(1.0, 0.5, 1.5, view_count=1),
        GuidanceMetrics(3.0, 1.5, 4.5, view_count=1),
    ]
    latest = progress.record_guidance(prefix=2, metrics=first_metrics)

    assert latest == GuidanceMetrics(2.0, 1.0, 3.0, view_count=2)
    assert progress.guidance_metrics_since_prefix == first_metrics
    assert load_json(tmp_path / "summary.json") == {
        "method": "dresg",
        "status": "running",
        "completed_guidance_steps": 1,
        "current_prefix": 2,
        "latest_guidance": {
            "style_loss": 2.0,
            "content_loss": 1.0,
            "total_loss": 3.0,
        },
        "rows": [],
    }

    second_metrics = [GuidanceMetrics(0.8, 0.4, 1.2, view_count=1)]
    progress.record_guidance(prefix=2, metrics=second_metrics)
    stage_metrics = {
        "prefix_length": 2,
        "timestep": 800,
        "teacher_l1": 0.2,
        "projection_gap_l1": 0.3,
        "post_color_transfer_enabled": 0,
    }
    row = progress.record_stage(stage_metrics)

    assert row == {
        "stage_index": 1,
        "prefix_length": 2,
        "timestep": 800,
        "guidance_style_loss": pytest.approx(1.6),
        "guidance_content_loss": pytest.approx(0.8),
        "teacher_l1": 0.2,
        "projection_gap_l1": 0.3,
        "post_color_transfer_enabled": 0,
    }
    assert progress.stage_rows == [row]
    assert progress.guidance_metrics_since_prefix == []
    assert load_json(tmp_path / "summary.json")["rows"] == [row]

    progress.update_final_stage({"post_color_transfer_enabled": 1})
    assert load_json(tmp_path / "summary.json")["rows"][-1][
        "post_color_transfer_enabled"
    ] == 1

    aggregate = {"num_stages": 1}
    parameter_stats = {"num_gaussians": 10}
    monkeypatch.setattr(
        progress,
        "_create_final_artifacts",
        lambda **_kwargs: {"video_elapsed_sec": 0.5},
    )
    monkeypatch.setattr(output_module, "build_aggregate", lambda _rows: aggregate)
    progress.finalize(
        scene=SimpleNamespace(parameter_stats=lambda: parameter_stats),
        source=object(),
        cameras=SimpleNamespace(
            c2w=torch.empty(0),
            view_indices=(1, 3),
        ),
        artifacts=object(),
        rendering=object(),
    )
    summary = load_json(tmp_path / "summary.json")
    assert summary["status"] == "complete"
    assert summary["aggregate"] == aggregate
    assert summary["final_parameter_stats"] == parameter_stats
    assert load_json(tmp_path / "aggregate_metrics.json") == aggregate
    assert summary["rows"][-1]["video_elapsed_sec"] == 0.5
    assert written_paths.count(tmp_path / "summary.json") == 7
    assert written_paths.count(tmp_path / "aggregate_metrics.json") == 1


def test_training_progress_weights_guidance_batches_by_view_count(tmp_path) -> None:
    progress = build_training_progress(torch.device("cpu"), tmp_path)

    latest = progress.record_guidance(
        prefix=1,
        metrics=[
            GuidanceMetrics(1.0, 2.0, 3.0, view_count=2),
            GuidanceMetrics(4.0, 5.0, 6.0, view_count=1),
        ],
    )

    assert latest.style_loss == pytest.approx(2.0)
    assert latest.content_loss == pytest.approx(3.0)
    assert latest.total_loss == pytest.approx(4.0)
    assert latest.view_count == 3


def test_training_progress_rejects_inconsistent_stage_prefix(tmp_path) -> None:
    progress = build_training_progress(torch.device("cpu"), tmp_path)
    progress.record_guidance(
        prefix=2,
        metrics=[GuidanceMetrics(1.0, 0.5, 1.5, view_count=1)],
    )

    with pytest.raises(ValueError, match="must match"):
        progress.record_stage({"prefix_length": 2})
