"""Teacher residual scales derived from the upstream diffusion timeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dresg.config import TeacherConfig


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _interpolate_scale(teacher: TeacherConfig, progress: float) -> float:
    return teacher.scale + progress * (teacher.gamma_max - teacher.scale)


def _scale_for_stage(
    *,
    teacher: TeacherConfig,
    alpha_bar: float,
    stage_index: int,
    stage_count: int,
) -> float:
    if teacher.mode == "constant":
        return teacher.scale
    if teacher.mode == "snr_balanced":
        progress = _clamp_unit(4.0 * alpha_bar * (1.0 - alpha_bar))
        return _interpolate_scale(teacher, progress)
    if teacher.mode == "snr_triangle":
        progress = _clamp_unit(1.0 - abs(2.0 * alpha_bar - 1.0))
        return _interpolate_scale(teacher, progress)
    if teacher.mode == "timestep_cosine":
        if stage_count == 1:
            progress = 0.0
        else:
            phase = math.pi * float(stage_index) / float(stage_count - 1)
            progress = math.sin(phase) ** 2
        return _interpolate_scale(teacher, progress)
    raise ValueError(f"Unsupported teacher mode: {teacher.mode}")


@dataclass(frozen=True, slots=True)
class TeacherScaleSchedule:
    """Precomputed teacher scales for the configured active stage prefixes."""

    prefixes: tuple[int, ...]
    timesteps: tuple[int, ...]
    alpha_bars: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.prefixes)
        if count == 0:
            raise ValueError("Teacher scale schedule must contain at least one stage")
        if not (
            count == len(self.timesteps) == len(self.alpha_bars) == len(self.scales)
        ):
            raise ValueError("Teacher scale schedule fields must have matching lengths")
        if self.prefixes != tuple(sorted(set(self.prefixes))):
            raise ValueError("Teacher scale prefixes must be unique, positive, and increasing")
        if self.prefixes[0] <= 0:
            raise ValueError("Teacher scale prefixes must be unique, positive, and increasing")
        if any(timestep < 0 for timestep in self.timesteps) or len(
            set(self.timesteps)
        ) != len(self.timesteps):
            raise ValueError("Teacher scale timesteps must be unique and non-negative")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.alpha_bars):
            raise ValueError("Teacher scale alpha_bar values must be finite and in [0, 1]")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scales):
            raise ValueError("Teacher scales must be finite and positive")

    @classmethod
    def from_timeline(
        cls,
        *,
        teacher: TeacherConfig,
        active_prefixes: Sequence[int],
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> TeacherScaleSchedule:
        prefixes = tuple(active_prefixes)
        if not prefixes:
            raise ValueError("Teacher scale schedule requires active prefixes")

        if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
            raise ValueError("Diffusion scheduler timesteps must be a one-dimensional Tensor")
        if timesteps.dtype == torch.bool or timesteps.is_floating_point() or timesteps.is_complex():
            raise ValueError("Diffusion scheduler timesteps must use an integer dtype")
        timestep_values = tuple(timesteps.detach().cpu().tolist())
        if not timestep_values:
            raise ValueError("Diffusion scheduler timesteps must not be empty")
        if len(set(timestep_values)) != len(timestep_values):
            raise ValueError("Diffusion scheduler timesteps must be unique")
        if max(prefixes) > len(timestep_values):
            raise ValueError(
                "Teacher scale prefix exceeds the diffusion timeline: "
                f"prefix={max(prefixes)} steps={len(timestep_values)}"
            )

        if not isinstance(alphas_cumprod, torch.Tensor) or alphas_cumprod.ndim != 1:
            raise ValueError(
                "Teacher scale schedule requires one-dimensional scheduler.alphas_cumprod"
            )
        alpha_values = tuple(alphas_cumprod.detach().float().cpu().tolist())
        if not alpha_values:
            raise ValueError("scheduler.alphas_cumprod must not be empty")
        if any(timestep < 0 or timestep >= len(alpha_values) for timestep in timestep_values):
            raise ValueError(
                "Diffusion scheduler timestep is outside scheduler.alphas_cumprod"
            )

        stage_timesteps = tuple(timestep_values[prefix - 1] for prefix in prefixes)
        stage_alphas = tuple(alpha_values[timestep] for timestep in stage_timesteps)
        stage_count = len(prefixes)
        scales = tuple(
            _scale_for_stage(
                teacher=teacher,
                alpha_bar=alpha_bar,
                stage_index=stage_index,
                stage_count=stage_count,
            )
            for stage_index, alpha_bar in enumerate(stage_alphas)
        )
        return cls(
            prefixes=prefixes,
            timesteps=stage_timesteps,
            alpha_bars=stage_alphas,
            scales=scales,
        )

    def scale_at(self, prefix: int) -> float:
        try:
            index = self.prefixes.index(prefix)
        except ValueError as error:
            raise ValueError(
                f"prefix={prefix} is absent from the teacher scale schedule"
            ) from error
        return self.scales[index]
