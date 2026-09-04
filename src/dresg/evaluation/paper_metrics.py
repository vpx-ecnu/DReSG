"""Paper metric orchestration and fixed-schema serialization."""

from __future__ import annotations

import csv
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dresg.utils.json_io import save_json

from .consistency import ConsistencyEvaluator, resolve_gap
from .features import (
    ClipEncoder,
    DinoEncoder,
    mean_cosine_to_reference,
    mean_pairwise_cosine,
)
from .layout import (
    MethodLayout,
    collect_method_layout,
    content_image_paths,
    layout_key,
)
from .performance import read_method_performance

MetricValue = int | float | str


def _require_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


PAPER_METRIC_COLUMNS = (
    "method",
    "scene",
    "style",
    "render_count",
    "clip_s",
    "dino_c",
    "st_lpips",
    "st_rmse",
    "lt_lpips",
    "lt_rmse",
    "train_time_sec",
    "train_peak_mem_mb",
    "infer_fps",
    "infer_peak_mem_mb",
)


@dataclass(frozen=True, slots=True)
class PaperMetricRow:
    method: str
    scene: str
    style: str
    render_count: int
    clip_s: float
    dino_c: float
    st_lpips: float
    st_rmse: float
    lt_lpips: float
    lt_rmse: float
    train_time_sec: float
    train_peak_mem_mb: float
    infer_fps: float
    infer_peak_mem_mb: float

    def to_ordered_dict(self) -> dict[str, MetricValue]:
        data = asdict(self)
        return {key: data[key] for key in PAPER_METRIC_COLUMNS}


def evaluate_paper_metrics(
    *,
    qual_root: Path,
    scene_dir: Path,
    scene: str,
    style: str,
    style_path: Path,
    methods: Sequence[str],
    factor: int,
    device: torch.device,
    batch_size: int,
    consistency_samples: int,
    short_gap: str,
    long_gap: str,
    view_mode: str,
    result_dirs: Mapping[str, Path],
    clip_model: str,
    dino_model: str,
    offline_models: bool,
) -> list[PaperMetricRow]:
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise TypeError("Methods must be a sequence of method names")
    if not isinstance(result_dirs, Mapping):
        raise TypeError("Result directories must be a method-to-path mapping")
    canonical_methods = tuple(layout_key(method, name="Method") for method in methods)
    if not canonical_methods:
        raise ValueError("Methods must not be empty")
    if len(set(canonical_methods)) != len(canonical_methods):
        raise ValueError("Methods must be unique")
    unknown_result_dirs = sorted(set(result_dirs) - set(canonical_methods))
    if unknown_result_dirs:
        raise ValueError(
            f"Result directories reference unknown methods: {unknown_result_dirs}"
        )
    scene = layout_key(scene, name="Scene")
    style = layout_key(style, name="Style")
    factor = _require_positive_integer(factor, name="Image factor")
    batch_size = _require_positive_integer(batch_size, name="Feature batch size")
    consistency_samples = _require_positive_integer(
        consistency_samples,
        name="Consistency sample count",
    )
    if view_mode not in {"all", "rendered"}:
        raise ValueError(f"Unsupported view_mode: {view_mode}")
    for gap_spec in (short_gap, long_gap):
        resolve_gap(gap_spec, 2)
    for model_name, name in ((clip_model, "CLIP model"), (dino_model, "DINO model")):
        if not isinstance(model_name, str):
            raise TypeError(f"{name} must be a string")
        if not model_name:
            raise ValueError(f"{name} must not be empty")
    if not isinstance(offline_models, bool):
        raise TypeError("Offline-model policy must be a boolean")
    if not isinstance(device, torch.device):
        raise TypeError("Evaluation device must be torch.device")
    if not isinstance(scene_dir, Path) or not isinstance(qual_root, Path):
        raise TypeError("Evaluation roots must be pathlib.Path values")
    if not isinstance(style_path, Path):
        raise TypeError("Style image path must be pathlib.Path")
    if not style_path.is_file():
        raise FileNotFoundError(f"Style image does not exist: {style_path}")
    invalid_result_paths = [
        method for method, path in result_dirs.items() if not isinstance(path, Path)
    ]
    if invalid_result_paths:
        raise TypeError(
            f"Result directories must be pathlib.Path values: {invalid_result_paths}"
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    all_content_paths = content_image_paths(scene_dir, factor)

    layouts: dict[str, MethodLayout] = {}
    performance: dict[str, dict[str, float]] = {}
    for method in canonical_methods:
        layout = collect_method_layout(
            qual_root=qual_root,
            method=method,
            scene=scene,
            style=style,
            all_content_paths=all_content_paths,
            view_mode=view_mode,
            run_root=result_dirs.get(method),
        )
        for gap_spec in (short_gap, long_gap):
            gap = resolve_gap(gap_spec, len(layout.render_paths))
            if gap >= len(layout.render_paths):
                raise ValueError(
                    f"Consistency gap {gap} has no valid pair among "
                    f"{len(layout.render_paths)} views"
                )
        layouts[method] = layout
        performance[method] = read_method_performance(layout.root)

    clip_encoder = ClipEncoder(
        model_name=clip_model,
        device=device,
        offline_models=offline_models,
    )
    clip_style_feat = clip_encoder.encode([style_path], batch_size=batch_size)
    dino_encoder = DinoEncoder(
        model_id=dino_model,
        device=device,
        offline_models=offline_models,
    )
    dino_content_feat = dino_encoder.encode(
        all_content_paths,
        batch_size=batch_size,
    )
    dino_content_index = {path: index for index, path in enumerate(all_content_paths)}
    consistency = ConsistencyEvaluator.load(
        device,
        offline_models=offline_models,
    )

    rows: list[PaperMetricRow] = []
    for method in canonical_methods:
        layout = layouts[method]
        perf = performance[method]
        clip_render_feat = clip_encoder.encode(
            layout.render_paths,
            batch_size=batch_size,
        )
        dino_render_feat = dino_encoder.encode(
            layout.render_paths,
            batch_size=batch_size,
        )
        aligned_content_feat = torch.stack(
            [dino_content_feat[dino_content_index[path]] for path in layout.content_paths],
            dim=0,
        )
        st_metrics = consistency.evaluate(
            layout.content_paths,
            layout.render_paths,
            gap=resolve_gap(short_gap, len(layout.render_paths)),
            samples=consistency_samples,
        )
        lt_metrics = consistency.evaluate(
            layout.content_paths,
            layout.render_paths,
            gap=resolve_gap(long_gap, len(layout.render_paths)),
            samples=consistency_samples,
        )
        rows.append(
            PaperMetricRow(
                method=method,
                scene=scene,
                style=style,
                render_count=len(layout.render_paths),
                clip_s=mean_cosine_to_reference(clip_render_feat, clip_style_feat),
                dino_c=mean_pairwise_cosine(dino_render_feat, aligned_content_feat),
                st_lpips=st_metrics.lpips,
                st_rmse=st_metrics.rmse,
                lt_lpips=lt_metrics.lpips,
                lt_rmse=lt_metrics.rmse,
                train_time_sec=perf["train_time_sec"],
                train_peak_mem_mb=perf["train_peak_mem_mb"],
                infer_fps=perf["infer_fps"],
                infer_peak_mem_mb=perf["infer_peak_mem_mb"],
            )
        )
    return rows


def _rows_to_dicts(rows: Sequence[PaperMetricRow]) -> list[dict[str, MetricValue]]:
    return [row.to_ordered_dict() for row in rows]


def _write_metric_csv(path: Path, rows: Sequence[PaperMetricRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PAPER_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(_rows_to_dicts(rows))


def _write_metric_json(path: Path, rows: Sequence[PaperMetricRow]) -> None:
    save_json(path, _rows_to_dicts(rows))


def write_metric_bundle(
    output_dir: Path,
    rows: Sequence[PaperMetricRow],
    config: Mapping[str, object],
) -> None:
    if not rows:
        raise ValueError("Evaluation metric bundle must contain at least one row")
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    temporary_dir.mkdir()
    try:
        _write_metric_csv(temporary_dir / "paper_metrics.csv", rows)
        _write_metric_json(temporary_dir / "paper_metrics.json", rows)
        save_json(temporary_dir / "paper_metrics_config.json", config)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
