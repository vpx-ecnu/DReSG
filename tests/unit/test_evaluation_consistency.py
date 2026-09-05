"""Regression tests for the original photo-flow/full-image LPIPS protocol."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dresg.evaluation import consistency
from dresg.utils.flow import pad_to_multiple, unpad, warp_with_mask


class ScalarDistance(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).square().mean().reshape(1, 1, 1, 1)


def evaluator(lpips_model: torch.nn.Module | None = None) -> consistency.ConsistencyEvaluator:
    return consistency.ConsistencyEvaluator(
        raft_model=torch.nn.Identity(),
        raft_transforms=object(),
        lpips_model=ScalarDistance() if lpips_model is None else lpips_model,
        device=torch.device("cpu"),
    )


def mock_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        consistency,
        "load_rgb_chw01",
        lambda path: torch.full((3, 8, 8), float(path.stem[-1])),
    )


def evaluate_pair(metric: consistency.ConsistencyEvaluator):
    return metric.evaluate(
        [Path("content0.png"), Path("content1.png")],
        [Path("stylized0.png"), Path("stylized1.png")],
        gap=1,
        samples=6,
    )


def test_lpips_loader_selects_original_scalar_alexnet(monkeypatch) -> None:
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return ScalarDistance()

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=load))
    model = consistency._load_lpips_model(torch.device("cpu"), offline_models=False)

    assert calls == [{"net": "alex", "spatial": False}]
    assert not model.training


@pytest.mark.parametrize("shift", [-1.0, 0.0, 0.5, 1.0])
def test_warp_direction_and_strict_boundary(shift: float) -> None:
    image = torch.arange(6, dtype=torch.float32).expand(1, 3, 5, 6)
    flow = torch.zeros(1, 2, 5, 6)
    flow[:, 0] = shift

    sampled, valid = warp_with_mask(image, flow)

    expected_valid = torch.zeros(1, 1, 5, 6)
    for y in range(5):
        for x in range(6):
            if 0 < y < 4 and 0 < x + shift < 5:
                expected_valid[:, :, y, x] = 1
                torch.testing.assert_close(sampled[0, :, y, x], torch.full((3,), x + shift))
    torch.testing.assert_close(valid, expected_valid, rtol=0, atol=0)


def test_raft_symmetric_replicate_padding_and_unpadding() -> None:
    image = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
    padded, pad = pad_to_multiple(image, 8)

    assert pad == (0, 1, 1, 2)
    assert padded.shape == (1, 1, 8, 8)
    torch.testing.assert_close(padded[0, 0, 0, :7], image[0, 0, 0])
    torch.testing.assert_close(unpad(padded, pad), image, rtol=0, atol=0)


def test_lpips_fills_invalid_pixels_before_full_image_distance(monkeypatch) -> None:
    mock_images(monkeypatch)
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: torch.zeros(1, 2, 8, 8))
    inputs = []

    class RecordingDistance(ScalarDistance):
        def forward(self, a, b):
            inputs.append((a.clone(), b.clone()))
            return super().forward(a, b)

    result = evaluate_pair(evaluator(RecordingDistance()))

    expected = torch.full((1, 3, 8, 8), -1.0)
    expected[:, :, 1:-1, 1:-1] = 1.0
    torch.testing.assert_close(inputs[0][0], expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs[0][1], torch.full_like(expected, -1), rtol=0, atol=0)
    assert result.rmse == 1.0
    assert result.lpips == 4 * 36 / 64  # Full image, not a valid-pixel average.


def test_forward_flow_samples_frame_one_back_into_frame_zero(monkeypatch) -> None:
    first = torch.arange(8, dtype=torch.float32).expand(3, 8, 8) / 8
    second = torch.roll(first, shifts=1, dims=-1)
    monkeypatch.setattr(
        consistency, "load_rgb_chw01", lambda path: first if path.stem.endswith("0") else second
    )
    forward = torch.zeros(1, 2, 8, 8)
    forward[:, 0] = 1
    flows = iter((forward, -forward))
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: next(flows))

    result = evaluate_pair(evaluator())

    assert result.lpips == pytest.approx(0, abs=1e-12)
    assert result.rmse == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("cause", ["out_of_bounds", "cycle_disagreement"])
def test_empty_valid_domain_fails_before_lpips(monkeypatch, cause) -> None:
    mock_images(monkeypatch)
    forward = torch.full((1, 2, 8, 8), 1000.0 if cause == "out_of_bounds" else 0.0)
    backward = torch.full_like(forward, 2.0)
    flows = iter((forward, backward))
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: next(flows))

    class NeverCalled(torch.nn.Module):
        def forward(self, *_args):
            pytest.fail("LPIPS must not run for an empty valid domain")

    with pytest.raises(ValueError, match=r"pair 0->1 .*no valid pixels"):
        evaluate_pair(evaluator(NeverCalled()))


def test_empty_pair_is_not_hidden_by_other_valid_pairs(monkeypatch) -> None:
    mock_images(monkeypatch)
    zero = torch.zeros(1, 2, 8, 8)
    flows = iter((zero, zero, zero + 1000, zero))
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: next(flows))

    with pytest.raises(ValueError, match=r"pair 1->2 .*no valid pixels"):
        evaluator().evaluate(
            [Path(f"content{i}.png") for i in range(3)],
            [Path(f"stylized{i}.png") for i in range(3)],
            gap=1,
            samples=6,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("direction", [0, 1])
def test_nonfinite_flow_is_rejected_even_at_an_invalid_border(monkeypatch, bad, direction) -> None:
    mock_images(monkeypatch)
    flows = [torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8)]
    flows[direction][0, 0, 0, 0] = bad
    flow_iter = iter(flows)
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: next(flow_iter))

    with pytest.raises(ValueError, match="non-finite optical flow"):
        evaluate_pair(evaluator())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_lpips_fails_instead_of_emitting_a_score(monkeypatch, bad) -> None:
    mock_images(monkeypatch)
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: torch.zeros(1, 2, 8, 8))

    class InvalidDistance(torch.nn.Module):
        def forward(self, *_args):
            return torch.tensor([[[[bad]]]])

    with pytest.raises(ValueError, match="non-finite metric"):
        evaluate_pair(evaluator(InvalidDistance()))


def test_spatial_lpips_map_is_not_accepted_as_full_image_score(monkeypatch) -> None:
    mock_images(monkeypatch)
    monkeypatch.setattr(consistency, "raft_flow", lambda *_args: torch.zeros(1, 2, 8, 8))

    class SpatialDistance(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).square().mean(dim=1, keepdim=True)

    with pytest.raises(ValueError, match="scalar, non-spatial LPIPS"):
        evaluate_pair(evaluator(SpatialDistance()))


@pytest.mark.parametrize(
    ("count", "gap", "starts"),
    [(20, 1, [0, 4, 7, 11, 14, 18]), (20, 10, [0, 2, 4, 5, 7, 9]),
     (25, 12, [0, 2, 5, 7, 10, 12])],
)
def test_uniform_pair_sampling_uses_original_photos(monkeypatch, count, gap, starts) -> None:
    calls = []

    def image(path):
        offset = 0.5 if path.stem.startswith("stylized") else 0
        return torch.full((3, 8, 8), int(path.stem.split("-")[1]) / 100 + offset)

    def flow(_model, _transforms, a, b, _device):
        # Stylized inputs would be >= 0.5 and must never reach RAFT.
        assert a.max() < 0.5 and b.max() < 0.5
        calls.append((round(a[0, 0, 0].item() * 100), round(b[0, 0, 0].item() * 100)))
        return torch.zeros(1, 2, 8, 8)

    monkeypatch.setattr(consistency, "load_rgb_chw01", image)
    monkeypatch.setattr(consistency, "raft_flow", flow)
    evaluator().evaluate(
        [Path(f"content-{i}.png") for i in range(count)],
        [Path(f"stylized-{i}.png") for i in range(count)],
        gap=gap,
        samples=6,
    )

    assert calls == [pair for i in starts for pair in ((i, i + gap), (i + gap, i))]


@pytest.mark.parametrize("field", ["gap", "samples"])
@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1])
def test_pair_request_requires_positive_canonical_integers(field, value) -> None:
    options = {"gap": 1, "samples": 6, field: value}
    error = ValueError if type(value) is int else TypeError
    with pytest.raises(error, match="Consistency"):
        evaluator().evaluate([Path("c0"), Path("c1")], [Path("s0"), Path("s1")], **options)
