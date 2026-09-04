from __future__ import annotations

import pytest
import torch

from dresg.config import TeacherConfig
from dresg.models.diffusion.scheduling.scale import TeacherScaleSchedule


class FakeScheduler:
    def __init__(
        self,
        *,
        timesteps: torch.Tensor | None = None,
        alphas_cumprod: torch.Tensor | None = None,
    ) -> None:
        self.timesteps = (
            torch.tensor([0, 1, 2], dtype=torch.long)
            if timesteps is None
            else timesteps
        )
        self.alphas_cumprod = (
            torch.tensor([0.01, 0.50, 0.99], dtype=torch.float32)
            if alphas_cumprod is None
            else alphas_cumprod
        )


def _teacher(
    mode: str,
    *,
    scale: float = 1.0,
    gamma_max: float = 3.0,
) -> TeacherConfig:
    return TeacherConfig(
        mode=mode,
        scale=scale,
        gamma_max=gamma_max,
    )


def _schedule(
    mode: str,
    *,
    prefixes: tuple[int, ...] = (1, 2, 3),
    scheduler: FakeScheduler | None = None,
) -> TeacherScaleSchedule:
    scheduler = FakeScheduler() if scheduler is None else scheduler
    return TeacherScaleSchedule.from_timeline(
        teacher=_teacher(mode),
        active_prefixes=prefixes,
        timesteps=scheduler.timesteps,
        alphas_cumprod=scheduler.alphas_cumprod,
    )


def test_teacher_scale_schedule_records_typed_diffusion_stage_data() -> None:
    schedule = _schedule("snr_balanced")

    assert schedule.prefixes == (1, 2, 3)
    assert schedule.timesteps == (0, 1, 2)
    assert schedule.alpha_bars == pytest.approx((0.01, 0.5, 0.99))
    assert schedule.scale_at(2) == pytest.approx(3.0)


def test_snr_balanced_scale_uses_normalized_signal_noise_product() -> None:
    schedule = _schedule("snr_balanced")

    expected_progress = tuple(4.0 * alpha * (1.0 - alpha) for alpha in schedule.alpha_bars)
    expected_scales = tuple(1.0 + progress * 2.0 for progress in expected_progress)
    assert schedule.scales == pytest.approx(expected_scales)
    assert schedule.scales[1] == pytest.approx(3.0)
    assert schedule.scales[0] == pytest.approx(schedule.scales[2])


def test_snr_triangle_is_parameter_free_and_peaks_at_half_signal() -> None:
    schedule = _schedule("snr_triangle")

    assert schedule.scales[1] == pytest.approx(3.0)
    assert schedule.scales[0] < schedule.scales[1]
    assert schedule.scales[2] < schedule.scales[1]
    assert schedule.scales[0] == pytest.approx(schedule.scales[2])


def test_timestep_cosine_uses_active_stage_rank() -> None:
    scheduler = FakeScheduler(
        timesteps=torch.arange(200, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 200),
    )
    schedule = _schedule(
        "timestep_cosine",
        prefixes=(10, 100, 200),
        scheduler=scheduler,
    )

    assert schedule.timesteps == (9, 99, 199)
    assert schedule.scales == pytest.approx((1.0, 3.0, 1.0))


def test_single_stage_cosine_and_constant_modes_use_base_scale() -> None:
    cosine = _schedule("timestep_cosine", prefixes=(2,))
    scheduler = FakeScheduler()
    constant = TeacherScaleSchedule.from_timeline(
        teacher=_teacher("constant", scale=2.0),
        active_prefixes=(1, 3),
        timesteps=scheduler.timesteps,
        alphas_cumprod=scheduler.alphas_cumprod,
    )

    assert cosine.scales == (1.0,)
    assert constant.scales == (2.0, 2.0)


def test_scale_lookup_rejects_inactive_prefix() -> None:
    schedule = _schedule("constant")

    with pytest.raises(ValueError, match="absent"):
        schedule.scale_at(4)


def test_schedule_rejects_prefix_beyond_diffusion_timeline() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _schedule("constant", prefixes=(4,))


def test_schedule_rejects_out_of_range_timestep_instead_of_clamping() -> None:
    scheduler = FakeScheduler(timesteps=torch.tensor([0, 1, 3]))

    with pytest.raises(ValueError, match="outside"):
        _schedule("constant", scheduler=scheduler)


def test_schedule_requires_integer_unique_timesteps_and_alpha_values() -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        _schedule(
            "constant",
            scheduler=FakeScheduler(timesteps=torch.tensor([0.0, 1.0, 2.0])),
        )
    with pytest.raises(ValueError, match="unique"):
        _schedule(
            "constant",
            scheduler=FakeScheduler(timesteps=torch.tensor([0, 1, 1])),
        )
    with pytest.raises(ValueError, match="alphas_cumprod"):
        _schedule(
            "constant",
            scheduler=FakeScheduler(alphas_cumprod=torch.ones(1, 3)),
        )


def test_schedule_rejects_unsupported_teacher_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported teacher mode"):
        _schedule("unsupported_schedule")
