from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from dresg.utils.view_selection import workflow
from dresg.utils.view_selection.support import SparseViewSupport


class FakeScene:
    def means(self) -> torch.Tensor:
        return torch.zeros((5, 3))


def test_view_selection_workflow_writes_result_and_hydra_preset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeSource:
        def __len__(self) -> int:
            return 3

    source = FakeSource()
    cameras = [
        SimpleNamespace(view_index=0),
        SimpleNamespace(view_index=1),
        SimpleNamespace(view_index=2),
    ]
    monkeypatch.setattr(workflow, "load_colmap_scene", lambda **_kwargs: source)
    monkeypatch.setattr(workflow, "build_scaled_cameras", lambda **_kwargs: cameras)
    monkeypatch.setattr(
        workflow,
        "build_gaussian_scene",
        lambda *_args, **_kwargs: FakeScene(),
    )
    monkeypatch.setattr(
        workflow,
        "compute_view_support",
        lambda camera, **_kwargs: SparseViewSupport(
            view_index=int(camera.view_index),
            gaussian_indices=torch.tensor([int(camera.view_index)]),
            values=torch.tensor([1.0]),
            visible_samples=1,
            depth_rejected=0,
        ),
    )
    monkeypatch.setattr(
        workflow,
        "greedy_select_views",
        lambda **_kwargs: [
            {
                "views": [0, 2],
                "covered_ratio": 0.8,
                "stop_reason": "min_marginal_gain_ratio",
            }
        ],
    )

    result = workflow.run_view_selection(
        workflow.ViewSelectionRequest(
            scene_dir=tmp_path / "scene",
            base_ply=tmp_path / "base.ply",
            output_dir=tmp_path / "selection",
            dataset="llff",
            scene="fern",
            device=torch.device("cpu"),
            factor=4,
            render_scale=1.0,
            candidate_views=(),
            seed_views=(),
            pool_grid_size=3,
            pool_radius_scale=1.0,
            depth_gate=True,
            depth_tolerance=0.05,
            depth_tolerance_ratio=0.01,
            min_weight=1e-4,
            target_fraction_of_max=0.98,
            min_marginal_gain_ratio=0.001,
            stop_coverage_ratio=0.9999,
            max_select=None,
        )
    )

    assert result["selected_views"] == [0, 2]
    saved = json.loads((tmp_path / "selection" / "coverage_selection.json").read_text())
    assert saved["candidate_views"] == [0, 1, 2]
    preset = yaml.safe_load((tmp_path / "selection" / "view_selection.yaml").read_text())
    assert preset["data"]["views"] == [0, 2]
    assert preset["view_selection"] == {
        "dataset": "llff",
        "scene": "fern",
        "candidate_count": 3,
        "selected_count": 2,
        "coverage_ratio": 0.8,
    }
