"""Active-view support estimation and greedy selection."""

from dresg.utils.view_selection.greedy import (
    coverage_stats,
    greedy_select_views,
)
from dresg.utils.view_selection.support import (
    SparseViewSupport,
    compute_view_support,
    sample_image_features,
)
from dresg.utils.view_selection.workflow import (
    ViewSelectionRequest,
    run_view_selection,
)

__all__ = [
    "SparseViewSupport",
    "ViewSelectionRequest",
    "compute_view_support",
    "coverage_stats",
    "greedy_select_views",
    "run_view_selection",
    "sample_image_features",
]
