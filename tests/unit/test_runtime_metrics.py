from __future__ import annotations

import time

import pytest
import torch

from dresg.training.output import _profile_optional_artifact
from dresg.utils.runtime_metrics import RuntimeMetricsTracker, RuntimeSectionProfiler


def test_runtime_section_profiler_cpu_reports_elapsed_and_zero_memory() -> None:
    profiler = RuntimeSectionProfiler(torch.device("cpu"))
    profiler.start()
    time.sleep(0.001)
    metrics = profiler.finish()

    assert metrics.elapsed_sec > 0.0
    assert metrics.peak_allocated_mb == 0.0


def test_runtime_tracker_cpu_stage_metrics_exclude_setup_and_reset_guidance_bucket() -> None:
    tracker = RuntimeMetricsTracker(torch.device("cpu"))

    guidance_profiler = tracker.profile_guidance_step_start()
    time.sleep(0.001)
    tracker.record_guidance_step(guidance_profiler.finish())

    tracker.start_stage()
    time.sleep(0.001)
    row = tracker.finish_stage(fit_peak_allocated_mb=0.0)

    assert row["guidance_elapsed_since_prefix_sec"] > 0.0
    assert row["stage_elapsed_sec"] > 0.0
    assert row["train_measured_elapsed_sec"] >= (row["guidance_elapsed_since_prefix_sec"] + row["stage_elapsed_sec"])
    assert row["peak_allocated_mb"] == 0.0

    tracker.start_stage()
    second = tracker.finish_stage(fit_peak_allocated_mb=0.0)
    assert second["guidance_elapsed_since_prefix_sec"] == pytest.approx(0.0)


def test_optional_artifact_profiler_skips_disabled_action() -> None:
    called = False

    def action() -> None:
        nonlocal called
        called = True

    metrics = _profile_optional_artifact(
        enabled=False,
        device=torch.device("cpu"),
        metric_prefix="video",
        action=action,
    )

    assert not called
    assert metrics == {
        "video_elapsed_sec": 0.0,
        "video_peak_allocated_mb": 0.0,
    }


def test_optional_artifact_profiler_measures_enabled_action() -> None:
    metrics = _profile_optional_artifact(
        enabled=True,
        device=torch.device("cpu"),
        metric_prefix="video",
        action=lambda: time.sleep(0.001),
    )

    assert metrics["video_elapsed_sec"] > 0.0
    assert metrics["video_peak_allocated_mb"] == 0.0
