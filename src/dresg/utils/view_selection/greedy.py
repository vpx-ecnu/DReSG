from __future__ import annotations

from collections.abc import Sequence

import torch

from dresg.utils.view_selection.support import SparseViewSupport


def _maximum_support(
    supports: Sequence[SparseViewSupport],
    *,
    gaussian_count: int,
) -> torch.Tensor:
    maximum = torch.zeros(gaussian_count, dtype=torch.float32)
    for support in supports:
        indices = support.gaussian_indices
        maximum[indices] = torch.maximum(
            maximum[indices],
            support.values.float(),
        )
    return maximum


def _apply_support_max(
    coverage: torch.Tensor,
    support: SparseViewSupport,
) -> None:
    indices = support.gaussian_indices
    coverage[indices] = torch.maximum(
        coverage[indices],
        support.values.float(),
    )


def coverage_stats(
    *,
    coverage: torch.Tensor,
    maximum_support: torch.Tensor,
    selected_views: Sequence[int],
    min_weight: float,
    target_fraction_of_max: float,
) -> dict[str, object]:
    target_visible = maximum_support >= min_weight
    target_support = (maximum_support * target_fraction_of_max).clamp_min(min_weight)
    covered = target_visible & (coverage >= target_support)
    relative = (coverage[target_visible] / target_support[target_visible].clamp_min(1e-6)).clamp(max=1.0)
    visible_count = target_visible.sum().item()
    covered_count = covered.sum().item()
    return {
        "views": list(selected_views),
        "num_views": len(selected_views),
        "target_visible_count": visible_count,
        "covered_count": covered_count,
        "covered_ratio": covered_count / max(1, visible_count),
        "mean_relative_coverage": relative.mean().item() if relative.numel() else 0.0,
        "min_relative_coverage": relative.min().item() if relative.numel() else 0.0,
    }


def greedy_select_views(
    *,
    supports: Sequence[SparseViewSupport],
    gaussian_count: int,
    seed_views: Sequence[int],
    min_weight: float,
    target_fraction_of_max: float,
    max_select: int | None,
    stop_coverage_ratio: float,
    min_marginal_gain_ratio: float,
) -> list[dict[str, object]]:
    support_by_view = {support.view_index: support for support in supports}
    if len(support_by_view) != len(supports):
        raise ValueError("Candidate support contains duplicate view indices")
    selected = list(seed_views)
    if len(set(selected)) != len(selected):
        raise ValueError("seed_views contains duplicates")
    missing_seeds = [view for view in selected if view not in support_by_view]
    if missing_seeds:
        raise ValueError(f"seed_views are not candidates: {missing_seeds}")

    selected_set = set(selected)
    remaining = [support.view_index for support in supports if support.view_index not in selected_set]
    maximum_support = _maximum_support(
        supports,
        gaussian_count=gaussian_count,
    )
    target_visible = maximum_support >= min_weight
    target_support = (maximum_support * target_fraction_of_max).clamp_min(min_weight)
    visible_count = target_visible.sum().item()
    coverage = torch.zeros_like(maximum_support)
    for view in selected:
        _apply_support_max(coverage, support_by_view[view])

    curve: list[dict[str, object]] = []
    while True:
        stats = coverage_stats(
            coverage=coverage,
            maximum_support=maximum_support,
            selected_views=selected,
            min_weight=min_weight,
            target_fraction_of_max=target_fraction_of_max,
        )
        curve.append(stats)
        if visible_count == 0:
            stats["stop_reason"] = "no_visible_gaussians"
            break
        if max_select is not None and len(selected) >= max_select:
            stats["stop_reason"] = "max_select"
            break
        if not remaining:
            stats["stop_reason"] = "no_remaining_views"
            break

        best_view: int | None = None
        best_gain = -1.0
        for view in remaining:
            support = support_by_view[view]
            indices = support.gaussian_indices
            valid = target_visible[indices]
            if valid.any():
                indices = indices[valid]
                values = support.values.float()[valid]
                before = (coverage[indices] / target_support[indices].clamp_min(1e-6)).clamp(max=1.0)
                after = (torch.maximum(coverage[indices], values) / target_support[indices].clamp_min(1e-6)).clamp(
                    max=1.0
                )
                gain = (after - before).sum().item()
            else:
                gain = 0.0
            if gain > best_gain:
                best_gain = gain
                best_view = view

        gain_ratio = best_gain / max(1, visible_count)
        stats["best_next_view"] = best_view
        stats["best_next_gain"] = best_gain
        stats["best_next_gain_ratio"] = gain_ratio
        if stats["covered_ratio"] >= stop_coverage_ratio:
            stats["stop_reason"] = "coverage_ratio"
            break
        if gain_ratio < min_marginal_gain_ratio:
            stats["stop_reason"] = "min_marginal_gain_ratio"
            break
        if best_view is None:
            stats["stop_reason"] = "no_remaining_views"
            break
        selected.append(best_view)
        remaining.remove(best_view)
        _apply_support_max(coverage, support_by_view[best_view])
    return curve
