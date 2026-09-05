"""Feature normalization, aggregation, and invalid-output contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from dresg.evaluation import features


@pytest.fixture(params=[features.ClipEncoder, features.DinoEncoder])
def encoder(request, monkeypatch):
    """Keep the production encode path, replacing only the model/image resources."""
    encoder_type = request.param
    instance = encoder_type.__new__(encoder_type)
    instance._device = torch.device("cpu")
    outputs = torch.tensor([[3.0, 4.0], [4.0, 3.0], [-3.0, -4.0]])
    monkeypatch.setattr(features, "load_pil_rgb", lambda path: int(path.stem))

    if encoder_type is features.ClipEncoder:
        instance._preprocess = lambda image: torch.tensor(image)
        instance._model = SimpleNamespace(encode_image=lambda batch: outputs[batch])
    else:
        instance._processor = lambda *, images, return_tensors: BatchFeature(
            {"pixel_values": torch.tensor(images)}
        )
        instance._model = lambda *, pixel_values: SimpleNamespace(pooler_output=outputs[pixel_values])
    return instance, outputs


def test_encoder_preserves_normalization_order_and_partial_batch(encoder) -> None:
    instance, raw = encoder
    result = instance.encode([Path(f"{i}.png") for i in range(3)], batch_size=2)

    torch.testing.assert_close(result, raw / raw.norm(dim=-1, keepdim=True), rtol=0, atol=0)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert not result.requires_grad


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0])
def test_encoder_rejects_nonfinite_or_zero_features(encoder, bad) -> None:
    instance, raw = encoder
    raw[1] = bad

    with pytest.raises(ValueError, match="features must"):
        instance.encode([Path(f"{i}.png") for i in range(3)], batch_size=2)


@pytest.mark.parametrize("batch_size", [True, 1.0, "1", 0, -1])
def test_encoder_rejects_invalid_batch_size(encoder, batch_size) -> None:
    instance, _ = encoder
    error = ValueError if type(batch_size) is int else TypeError
    with pytest.raises(error, match="Feature batch size"):
        instance.encode([Path("0.png")], batch_size=batch_size)


def test_encoder_rejects_empty_images(encoder) -> None:
    instance, _ = encoder
    with pytest.raises(ValueError, match="at least one image"):
        instance.encode([], batch_size=2)


def test_cosine_aggregation_matches_original_style_and_aligned_view_formulas() -> None:
    rendered = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    style = torch.tensor([[1.0, 0.0]])
    content = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    assert features.mean_cosine_to_reference(rendered, style) == pytest.approx(1.6 / 3)
    assert features.mean_pairwise_cosine(rendered, content) == pytest.approx(1.8 / 3)
    # Pairwise alignment is not all-to-all matching or a cosine of average features.
    assert features.mean_pairwise_cosine(rendered, content.flip(0)) != pytest.approx(1.8 / 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_cosine_aggregation_rejects_nonfinite_scores(bad) -> None:
    invalid = torch.tensor([[bad, 1.0]])
    reference = torch.ones(1, 2)
    for metric in (features.mean_cosine_to_reference, features.mean_pairwise_cosine):
        with pytest.raises(ValueError, match="Cosine similarity must be finite"):
            metric(invalid, reference)


def test_cosine_aggregation_rejects_empty_scores() -> None:
    empty = torch.empty(0, 2)
    with pytest.raises(ValueError, match="non-empty"):
        features.mean_cosine_to_reference(empty, torch.ones(1, 2))
    with pytest.raises(ValueError, match="non-empty"):
        features.mean_pairwise_cosine(empty, empty)


def test_default_quality_backbones_match_original_evaluation() -> None:
    assert features.DEFAULT_CLIP_MODEL == "ViT-B/32"
    assert features.DEFAULT_DINO_MODEL == "facebook/dinov2-base"
