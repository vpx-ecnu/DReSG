import json
from pathlib import Path

import pytest

from dresg.evaluation.performance import read_method_performance


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_method_performance_uses_canonical_artifacts(tmp_path: Path) -> None:
    write_json(
        tmp_path / "export_metrics.json",
        {"infer_fps": 12.5, "infer_peak_mem_mb": 321},
    )
    write_json(
        tmp_path / "aggregate_metrics.json",
        {"train_measured_elapsed_sec": 7.0, "peak_allocated_mb": 654},
    )

    assert read_method_performance(tmp_path) == {
        "train_time_sec": 7.0,
        "train_peak_mem_mb": 654,
        "infer_fps": 12.5,
        "infer_peak_mem_mb": 321,
    }


def test_read_method_performance_rejects_non_numeric_canonical_value(tmp_path: Path) -> None:
    write_json(
        tmp_path / "export_metrics.json",
        {"infer_fps": "fast", "infer_peak_mem_mb": 321},
    )
    write_json(
        tmp_path / "aggregate_metrics.json",
        {"train_measured_elapsed_sec": 7.0, "peak_allocated_mb": 654},
    )

    with pytest.raises(ValueError, match="infer_fps must be numeric"):
        read_method_performance(tmp_path)


def test_read_method_performance_rejects_legacy_only_artifacts(tmp_path: Path) -> None:
    write_json(
        tmp_path / "export_metrics.json",
        {"infer_peak_allocated_mb": 100},
    )
    write_json(
        tmp_path / "aggregate_metrics.json",
        {"elapsed_sec": 11, "peak_gpu_mem_mb": 222},
    )

    with pytest.raises(ValueError, match="missing required field"):
        read_method_performance(tmp_path)


def test_read_method_performance_requires_both_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing metric artifact"):
        read_method_performance(tmp_path)
