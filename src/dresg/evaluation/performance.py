"""Read the canonical performance artifacts written by DReSG."""

from __future__ import annotations

from pathlib import Path

from dresg.utils.json_io import load_json
from dresg.utils.results import export_metrics_path


def _load_metric_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metric artifact: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Metric artifact must contain a JSON object: {path}")
    return payload


def _required_number(
    payload: dict[str, object],
    key: str,
    *,
    source: Path,
) -> float:
    if key not in payload:
        raise ValueError(f"{source} is missing required field {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} field {key} must be numeric")
    return float(value)


def read_method_performance(root: Path) -> dict[str, float]:
    export_path = export_metrics_path(root)
    aggregate_path = root / "aggregate_metrics.json"
    export = _load_metric_object(export_path)
    aggregate = _load_metric_object(aggregate_path)
    return {
        "train_time_sec": _required_number(
            aggregate,
            "train_measured_elapsed_sec",
            source=aggregate_path,
        ),
        "train_peak_mem_mb": _required_number(
            aggregate,
            "peak_allocated_mb",
            source=aggregate_path,
        ),
        "infer_fps": _required_number(export, "infer_fps", source=export_path),
        "infer_peak_mem_mb": _required_number(
            export,
            "infer_peak_mem_mb",
            source=export_path,
        ),
    }
