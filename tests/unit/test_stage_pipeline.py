from __future__ import annotations

from types import SimpleNamespace

import torch

from dresg.training import stages as stages_package
from dresg.training.output import build_training_progress
from dresg.training.stages import ColorStage, GuidanceStage
from dresg.training.stages import guidance as guidance_stage
from tests.config_factory import make_config


def test_guidance_stage_returns_view_metrics_and_profiles_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    progress = build_training_progress(torch.device("cpu"), tmp_path)
    config = make_config()
    config.guidance.optimization.view_batch_size = 2
    prepared_batches: list[tuple[int, ...]] = []
    timestep_state = object()

    def prepare_batch(_timestep_state, view_ids):
        ids = tuple(view_ids)
        prepared_batches.append(ids)
        return SimpleNamespace(view_ids=ids)

    guidance = SimpleNamespace(
        view_ids=(4, 7, 9),
        prepare_timestep=lambda _timestep: timestep_state,
        prepare_batch=prepare_batch,
    )

    class FakeOptimizer:
        def __init__(self, _guidance, batch, _optimization) -> None:
            self._batch = batch

        def run(self):
            return SimpleNamespace(
                style_loss=1.0,
                content_loss=0.5,
                total_loss=float(self._batch.view_ids[0]),
                view_count=len(self._batch.view_ids),
            )

    monkeypatch.setattr(guidance_stage, "GuidanceBatchOptimizer", FakeOptimizer)
    metrics = GuidanceStage(
        guidance,
        config.guidance.optimization,
        progress.runtime_metrics,
    ).run(torch.tensor(900))

    assert prepared_batches == [(4, 7), (9,)]
    assert [metric.total_loss for metric in metrics] == [4.0, 9.0]
    assert [metric.view_count for metric in metrics] == [2, 1]
    assert progress.guidance_metrics_since_prefix == []
    assert progress.runtime_metrics.train_measured_elapsed_sec > 0.0


def test_stage_package_exports_only_execution_boundaries() -> None:
    assert stages_package.__all__ == (
        "ColorStage",
        "FeedbackStage",
        "GuidanceStage",
    )
    assert callable(ColorStage)


class TestDebugConfig:
    def test_stage_diagnostics_use_one_boolean_flag(self) -> None:
        cfg = make_config()
        assert cfg.debug.collect_stage_diagnostics is False
        assert not hasattr(cfg.debug, "stage_metrics_mode")


def test_schedule_active_prefixes_applies_max_stages_truncation() -> None:
    cfg = make_config()
    cfg.schedule.prefixes = [10, 20, 30]
    cfg.schedule.max_stages = 2

    assert cfg.schedule.active_prefixes == (10, 20)
