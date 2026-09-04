from __future__ import annotations

import pytest
import torch

from dresg.utils.tensor_stats import average_metric_dicts, image_color_stats, tensor_range_stats


def test_tensor_range_stats_reduce_one_tensor() -> None:
    stats = tensor_range_stats("value", torch.tensor([-1.0, 0.5, 2.0]))

    assert stats == {
        "value_min": -1.0,
        "value_max": 2.0,
        "value_under0_frac": pytest.approx(1.0 / 3.0),
        "value_over1_frac": pytest.approx(1.0 / 3.0),
    }


def test_image_color_stats_report_rgb_means() -> None:
    image = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.5, 0.0]],
            [[0.0, 0.0]],
        ]
    )

    stats = image_color_stats("image", image)

    assert stats["image_value_mean"] == pytest.approx(0.5)
    assert stats["image_chroma_mean"] == pytest.approx(0.5)
    assert stats["image_clip_hi_frac"] == pytest.approx(1.0 / 6.0)
    assert stats["image_clip_lo_frac"] == pytest.approx(4.0 / 6.0)


def test_average_metric_dicts_preserves_metric_keys() -> None:
    assert average_metric_dicts([{"a": 1.0}, {"a": 3.0}]) == {"a": 2.0}
