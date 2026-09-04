"""Wall-time and CUDA memory helpers for measured training sections."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

MB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class RuntimeMemorySnapshot:
    allocated_mb: float = 0.0
    peak_allocated_mb: float = 0.0


@dataclass(frozen=True, slots=True)
class RuntimeSectionMetrics:
    elapsed_sec: float = 0.0
    peak_allocated_mb: float = 0.0


def is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def synchronize_device(device: torch.device) -> None:
    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device: torch.device) -> RuntimeMemorySnapshot:
    if not is_cuda_device(device):
        return RuntimeMemorySnapshot()
    return RuntimeMemorySnapshot(
        allocated_mb=torch.cuda.memory_allocated(device) / MB,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / MB,
    )


class RuntimeSectionProfiler:
    """Measure one synchronous section using wall time and CUDA peak memory."""

    def __init__(self, device: torch.device, *, reset_peak: bool = True) -> None:
        self.device = device
        self.reset_peak = reset_peak
        self._start: float | None = None

    def start(self) -> None:
        synchronize_device(self.device)
        if self.reset_peak:
            reset_peak_memory(self.device)
        synchronize_device(self.device)
        self._start = time.perf_counter()

    def finish(self) -> RuntimeSectionMetrics:
        if self._start is None:
            raise RuntimeError("RuntimeSectionProfiler.finish() called before start()")
        synchronize_device(self.device)
        elapsed = time.perf_counter() - self._start
        self._start = None
        return RuntimeSectionMetrics(
            elapsed_sec=elapsed,
            peak_allocated_mb=memory_snapshot(self.device).peak_allocated_mb,
        )


class RuntimeMetricsTracker:
    """Accumulate measured guidance and stage runtime after trainer setup."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        synchronize_device(device)
        baseline_allocated_mb = memory_snapshot(device).allocated_mb
        self.train_measured_elapsed_sec = 0.0
        self._global_peak_allocated_mb = baseline_allocated_mb
        self._guidance_elapsed_since_prefix_sec = 0.0
        self._stage_profiler: RuntimeSectionProfiler | None = None
        self._stage_pre_fit_peak_allocated_mb = baseline_allocated_mb

    def profile_guidance_step_start(self) -> RuntimeSectionProfiler:
        profiler = RuntimeSectionProfiler(self.device)
        profiler.start()
        return profiler

    def record_guidance_step(self, metrics: RuntimeSectionMetrics) -> None:
        self._guidance_elapsed_since_prefix_sec += metrics.elapsed_sec
        self.train_measured_elapsed_sec += metrics.elapsed_sec
        self._global_peak_allocated_mb = max(
            self._global_peak_allocated_mb,
            metrics.peak_allocated_mb,
        )

    def start_stage(self) -> None:
        self._stage_profiler = RuntimeSectionProfiler(self.device)
        self._stage_profiler.start()
        snapshot = memory_snapshot(self.device)
        self._stage_pre_fit_peak_allocated_mb = max(
            snapshot.allocated_mb,
            snapshot.peak_allocated_mb,
        )

    def capture_stage_peak(self) -> None:
        snapshot = memory_snapshot(self.device)
        self._stage_pre_fit_peak_allocated_mb = max(
            self._stage_pre_fit_peak_allocated_mb,
            snapshot.allocated_mb,
            snapshot.peak_allocated_mb,
        )

    def finish_stage(self, *, fit_peak_allocated_mb: float) -> dict[str, float]:
        if self._stage_profiler is None:
            raise RuntimeError("finish_stage() called before start_stage()")
        self.capture_stage_peak()
        stage_metrics = self._stage_profiler.finish()
        self._stage_profiler = None
        stage_peak_allocated = max(
            stage_metrics.peak_allocated_mb,
            self._stage_pre_fit_peak_allocated_mb,
            fit_peak_allocated_mb,
        )
        self.train_measured_elapsed_sec += stage_metrics.elapsed_sec
        self._global_peak_allocated_mb = max(
            self._global_peak_allocated_mb,
            stage_peak_allocated,
        )
        row = {
            "guidance_elapsed_since_prefix_sec": self._guidance_elapsed_since_prefix_sec,
            "stage_elapsed_sec": stage_metrics.elapsed_sec,
            "train_measured_elapsed_sec": self.train_measured_elapsed_sec,
            "peak_allocated_mb": self._global_peak_allocated_mb,
        }
        self._guidance_elapsed_since_prefix_sec = 0.0
        return row
