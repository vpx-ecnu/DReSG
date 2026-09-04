from __future__ import annotations

import torch

from dresg.utils.view_selection import (
    SparseViewSupport,
    greedy_select_views,
    sample_image_features,
)


def sparse(view: int, indices: list[int], values: list[float]) -> SparseViewSupport:
    return SparseViewSupport(
        view_index=view,
        gaussian_indices=torch.tensor(indices, dtype=torch.long),
        values=torch.tensor(values, dtype=torch.float32),
        visible_samples=len(indices),
        depth_rejected=0,
    )


def test_sample_image_features_uses_pixel_coordinates() -> None:
    image = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    sampled = sample_image_features(image, coords, width=2, height=2)

    assert sampled.shape == (2, 1)
    assert torch.allclose(sampled[:, 0], torch.tensor([0.0, 3.0]))


def test_greedy_selection_uses_gain_and_stops_below_threshold() -> None:
    supports = [
        sparse(10, [0], [1.0]),
        sparse(20, [1], [1.0]),
        sparse(30, [0, 1], [0.6, 0.6]),
    ]

    curve = greedy_select_views(
        supports=supports,
        gaussian_count=2,
        seed_views=(),
        min_weight=1e-4,
        target_fraction_of_max=0.98,
        max_select=None,
        min_marginal_gain_ratio=0.5,
        stop_coverage_ratio=0.9999,
    )

    assert curve[-1]["views"] == [30]
    assert curve[-1]["stop_reason"] == "min_marginal_gain_ratio"
    assert curve[0]["best_next_view"] == 30


def test_greedy_selection_respects_no_fixed_view_cap() -> None:
    supports = [
        sparse(0, [0], [1.0]),
        sparse(1, [1], [1.0]),
        sparse(2, [2], [1.0]),
    ]

    curve = greedy_select_views(
        supports=supports,
        gaussian_count=3,
        seed_views=(),
        min_weight=1e-4,
        target_fraction_of_max=0.98,
        max_select=None,
        min_marginal_gain_ratio=0.0,
        stop_coverage_ratio=1.0,
    )

    assert curve[-1]["views"] == [0, 1, 2]
    assert curve[-1]["covered_ratio"] == 1.0
