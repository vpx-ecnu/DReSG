from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from dresg.evaluation import paper_metrics as paper_metrics_module
from dresg.evaluation.consistency import (
    ConsistencyEvaluator,
    ConsistencyMetrics,
    _load_lpips_model,
    resolve_gap,
)
from dresg.evaluation.features import mean_pairwise_cosine
from dresg.evaluation.layout import MethodLayout
from dresg.evaluation.paper_metrics import (
    PAPER_METRIC_COLUMNS,
    PaperMetricRow,
    evaluate_paper_metrics,
    write_metric_bundle,
)
from dresg.utils.results import ViewManifestEntry, write_view_manifest


def old_metric_name(*parts: str) -> str:
    return "_".join(parts)


FORBIDDEN_METRIC_COLUMNS = {
    old_metric_name("csd", "s"),
    old_metric_name("dino", "s"),
    old_metric_name("clip", "c"),
    old_metric_name("dino", "c", "patch"),
    old_metric_name("dino", "c", "selfsim"),
    old_metric_name("content", "ssim"),
    old_metric_name("content", "lpips"),
    old_metric_name("content", "l1"),
    "status",
    "error",
}


def metric_row() -> PaperMetricRow:
    return PaperMetricRow(
        method="ours",
        scene="fern",
        style="018",
        render_count=2,
        clip_s=0.5,
        dino_c=0.6,
        st_lpips=0.1,
        st_rmse=0.01,
        lt_lpips=0.2,
        lt_rmse=0.02,
        train_time_sec=5.0,
        train_peak_mem_mb=456.0,
        infer_fps=10.0,
        infer_peak_mem_mb=123.0,
    )


def test_pairwise_cosine_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        mean_pairwise_cosine(torch.ones((2, 3)), torch.ones((1, 3)))


def test_paper_metric_schema_has_only_complete_columns() -> None:
    data = metric_row().to_ordered_dict()

    assert tuple(data) == PAPER_METRIC_COLUMNS
    assert not FORBIDDEN_METRIC_COLUMNS.intersection(data)
    assert all(value != "" for value in data.values())


def test_metric_bundle_csv_uses_fixed_column_order(tmp_path: Path) -> None:
    output_dir = tmp_path / "metrics"
    write_metric_bundle(output_dir, [metric_row()], {"scene": "fern"})

    with (output_dir / "paper_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == PAPER_METRIC_COLUMNS
        rows = list(reader)
    assert rows[0]["method"] == "ours"
    assert "csd_s" not in rows[0]


def test_short_consistency_sequence_fails_before_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        "dresg.evaluation.consistency.raft_flow",
        lambda *_args, **_kwargs: pytest.fail("RAFT flow must not run"),
    )
    evaluator = ConsistencyEvaluator(
        raft_model=torch.nn.Identity(),
        raft_transforms=object(),
        lpips_model=torch.nn.Identity(),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="has no valid pair"):
        evaluator.evaluate(
            [Path("content0.png")],
            [Path("stylized0.png")],
            gap=1,
            samples=6,
        )


def test_offline_lpips_requires_cached_alexnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Offline LPIPS AlexNet checkpoint"):
        _load_lpips_model(torch.device("cpu"), offline_models=True)


def test_resolve_gap_requires_canonical_spec_and_count() -> None:
    assert resolve_gap("n/2", 9) == 4
    assert resolve_gap("1", 9) == 1
    for invalid in ("half", "N/2", " 1", "01", "0", "-1"):
        with pytest.raises(ValueError):
            resolve_gap(invalid, 9)
    with pytest.raises(TypeError, match="view count must be an integer"):
        resolve_gap("1", True)


def test_evaluation_config_fails_before_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style_path = tmp_path / "style.png"
    style_path.touch()
    monkeypatch.setattr(
        paper_metrics_module,
        "ClipEncoder",
        lambda **_kwargs: pytest.fail("CLIP must not load"),
    )

    with pytest.raises(ValueError, match="Consistency sample count must be positive"):
        evaluate_paper_metrics(
            qual_root=tmp_path,
            scene_dir=tmp_path,
            scene="fern",
            style="018",
            style_path=style_path,
            methods=("ours",),
            factor=4,
            device=torch.device("cpu"),
            batch_size=16,
            consistency_samples=0,
            short_gap="1",
            long_gap="n/2",
            view_mode="all",
            result_dirs={},
            clip_model="ViT-B/32",
            dino_model="facebook/dinov2-base",
            offline_models=True,
        )


def test_artifact_preflight_fails_before_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style_path = tmp_path / "style.png"
    style_path.touch()
    content_paths = (tmp_path / "content0.png", tmp_path / "content1.png")
    layout = MethodLayout(
        root=tmp_path / "run",
        render_paths=(tmp_path / "render0.png", tmp_path / "render1.png"),
        content_paths=content_paths,
        view_indices=(0, 1),
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "content_image_paths",
        lambda *_args, **_kwargs: content_paths,
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "collect_method_layout",
        lambda **_kwargs: layout,
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "read_method_performance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing performance")
        ),
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "ClipEncoder",
        lambda **_kwargs: pytest.fail("CLIP must not load"),
    )

    with pytest.raises(FileNotFoundError, match="missing performance"):
        evaluate_paper_metrics(
            qual_root=tmp_path,
            scene_dir=tmp_path,
            scene="fern",
            style="018",
            style_path=style_path,
            methods=("ours",),
            factor=4,
            device=torch.device("cpu"),
            batch_size=16,
            consistency_samples=1,
            short_gap="1",
            long_gap="n/2",
            view_mode="all",
            result_dirs={},
            clip_model="ViT-B/32",
            dino_model="facebook/dinov2-base",
            offline_models=True,
        )


def test_consistency_rejects_mismatched_view_counts() -> None:
    evaluator = ConsistencyEvaluator(
        raft_model=torch.nn.Identity(),
        raft_transforms=object(),
        lpips_model=torch.nn.Identity(),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="equal content and stylized view counts"):
        evaluator.evaluate(
            [Path("content0.png"), Path("content1.png")],
            [Path("stylized0.png")],
            gap=1,
            samples=6,
        )


def test_consistency_evaluates_the_single_available_pair(monkeypatch) -> None:
    class SpatialLpips(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a - b).square().mean(dim=1, keepdim=True)

    monkeypatch.setattr(
        "dresg.evaluation.consistency.raft_flow",
        lambda _model, _transforms, a, _b, _device: torch.zeros(
            (1, 2, *a.shape[-2:]),
            dtype=a.dtype,
        ),
    )
    monkeypatch.setattr(
        "dresg.evaluation.consistency.load_rgb_chw01",
        lambda path: torch.zeros(3, 8, 8) if "0" in path.stem else torch.ones(3, 8, 8),
    )

    evaluator = ConsistencyEvaluator(
        raft_model=torch.nn.Identity(),
        raft_transforms=object(),
        lpips_model=SpatialLpips(),
        device=torch.device("cpu"),
    )
    metrics = evaluator.evaluate(
        [Path("content0.png"), Path("content1.png")],
        [Path("stylized0.png"), Path("stylized1.png")],
        gap=1,
        samples=6,
    )

    assert metrics.rmse == 1.0


def test_evaluate_paper_metrics_with_mocked_models(tmp_path: Path, monkeypatch) -> None:
    qual_root = tmp_path / "qualitative"
    run_root = qual_root / "ours" / "fern" / "018"
    renders = run_root / "renders"
    renders.mkdir(parents=True)
    (renders / "000000.png").touch()
    (renders / "000001.png").touch()
    write_view_manifest(
        run_root,
        [
            ViewManifestEntry(render_file="000000.png", view_index=0),
            ViewManifestEntry(render_file="000001.png", view_index=1),
        ],
    )
    (run_root / "export_metrics.json").write_text(
        json.dumps({"infer_fps": 10.0, "infer_peak_mem_mb": 123}), encoding="utf-8"
    )
    (run_root / "aggregate_metrics.json").write_text(
        json.dumps({"train_measured_elapsed_sec": 5.0, "peak_allocated_mb": 456}), encoding="utf-8"
    )

    content_paths = [tmp_path / "content0.png", tmp_path / "content1.png"]
    style_path = tmp_path / "018.png"
    style_path.touch()
    monkeypatch.setattr(
        paper_metrics_module,
        "content_image_paths",
        lambda *_args, **_kwargs: content_paths,
    )
    class FakeEncoder:
        def __init__(self, **_kwargs) -> None:
            pass

        def encode(self, paths, *, batch_size):
            return torch.tensor([[1.0, 0.0] for _ in paths])

    class FakeConsistencyEvaluator:
        @classmethod
        def load(cls, *_args, **_kwargs):
            return cls()

        def evaluate(self, *_args, **_kwargs):
            return ConsistencyMetrics(
                lpips=0.1,
                rmse=0.01,
            )

    monkeypatch.setattr(paper_metrics_module, "ClipEncoder", FakeEncoder)
    monkeypatch.setattr(paper_metrics_module, "DinoEncoder", FakeEncoder)
    monkeypatch.setattr(
        paper_metrics_module,
        "ConsistencyEvaluator",
        FakeConsistencyEvaluator,
    )

    [row] = evaluate_paper_metrics(
        qual_root=qual_root,
        scene_dir=tmp_path / "scene",
        scene="fern",
        style="018",
        style_path=style_path,
        methods=["ours"],
        factor=4,
        device=torch.device("cpu"),
        batch_size=16,
        consistency_samples=6,
        short_gap="1",
        long_gap="n/2",
        view_mode="all",
        result_dirs={},
        clip_model="ViT-B/32",
        dino_model="facebook/dinov2-base",
        offline_models=True,
    )

    result = row.to_ordered_dict()
    assert tuple(result) == PAPER_METRIC_COLUMNS
    assert result["clip_s"] == 1.0
    assert result["dino_c"] == 1.0
    assert result["st_lpips"] == 0.1
    assert result["lt_rmse"] == 0.01
    assert result["train_time_sec"] == 5.0
    assert result["infer_fps"] == 10.0
    assert not FORBIDDEN_METRIC_COLUMNS.intersection(result)


def test_metric_failure_propagates_without_partial_row(tmp_path: Path, monkeypatch) -> None:
    style_path = tmp_path / "018.png"
    style_path.touch()
    content_paths = (tmp_path / "content0.png", tmp_path / "content1.png")
    layout = MethodLayout(
        root=tmp_path / "run",
        render_paths=(tmp_path / "render0.png", tmp_path / "render1.png"),
        content_paths=content_paths,
        view_indices=(0, 1),
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "content_image_paths",
        lambda *_args, **_kwargs: content_paths,
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "collect_method_layout",
        lambda **_kwargs: layout,
    )
    monkeypatch.setattr(
        paper_metrics_module,
        "read_method_performance",
        lambda *_args, **_kwargs: {},
    )

    class FailingClipEncoder:
        def __init__(self, **_kwargs) -> None:
            pass

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("clip failed")

    monkeypatch.setattr(paper_metrics_module, "ClipEncoder", FailingClipEncoder)

    with pytest.raises(RuntimeError, match="clip failed"):
        evaluate_paper_metrics(
            qual_root=tmp_path / "qualitative",
            scene_dir=tmp_path,
            scene="fern",
            style="018",
            style_path=style_path,
            methods=["ours"],
            factor=4,
            device=torch.device("cpu"),
            batch_size=16,
            consistency_samples=6,
            short_gap="1",
            long_gap="n/2",
            view_mode="all",
            result_dirs={},
            clip_model="ViT-B/32",
            dino_model="facebook/dinov2-base",
            offline_models=True,
        )


def test_metric_bundle_commits_all_outputs_together(tmp_path: Path) -> None:
    output_dir = tmp_path / "metrics"

    write_metric_bundle(output_dir, [metric_row()], {"scene": "fern"})

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "paper_metrics.csv",
        "paper_metrics.json",
        "paper_metrics_config.json",
    ]


def test_metric_bundle_rejects_empty_rows_without_creating_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "metrics"

    with pytest.raises(ValueError, match="at least one row"):
        write_metric_bundle(output_dir, [], {"scene": "fern"})

    assert not output_dir.exists()


def test_metric_bundle_failure_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "metrics"
    monkeypatch.setattr(
        paper_metrics_module,
        "_write_metric_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        write_metric_bundle(output_dir, [metric_row()], {"scene": "fern"})

    assert not output_dir.exists()
    assert not (tmp_path / ".metrics.tmp").exists()
