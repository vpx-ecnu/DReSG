from __future__ import annotations

import gc
import sys
import weakref
from types import SimpleNamespace

import torch

from dresg.models.gs.fitting.dino import DinoPatchContentLoss


class FakeDinoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, *, pixel_values: torch.Tensor):
        batch = int(pixel_values.shape[0])
        tokens = pixel_values.mean(dim=(-2, -1)).unsqueeze(1).repeat(1, 2, 1)
        cls_token = torch.zeros(batch, 1, tokens.shape[-1], device=tokens.device)
        return SimpleNamespace(last_hidden_state=torch.cat([cls_token, tokens], dim=1))


class FakeAutoModel:
    calls: list[tuple[str, bool]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, *, local_files_only: bool):
        cls.calls.append((model_name, local_files_only))
        return FakeDinoModel()


def _content_loss(*, local_files_only: bool = False) -> DinoPatchContentLoss:
    return DinoPatchContentLoss(
        model_name="example/dino",
        size=28,
        local_files_only=local_files_only,
        device=torch.device("cpu"),
    )


def test_dino_loader_respects_run_mode_and_loads_once(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=FakeAutoModel),
    )
    FakeAutoModel.calls.clear()
    online = _content_loss(local_files_only=False)
    offline = _content_loss(local_files_only=True)

    assert online._load_model() is online._load_model()
    offline._load_model()

    assert FakeAutoModel.calls == [
        ("example/dino", False),
        ("example/dino", True),
    ]


def test_dino_base_tokens_are_cached_by_view(monkeypatch) -> None:
    content_loss = _content_loss()
    calls: list[torch.Tensor] = []

    def fake_tokens(image_bchw: torch.Tensor) -> torch.Tensor:
        calls.append(image_bchw)
        return image_bchw.new_zeros((1, 2, 3))

    monkeypatch.setattr(content_loss, "_patch_tokens", fake_tokens)
    base_render = torch.zeros((1, 3, 28, 28))

    first = content_loss._base_tokens(view_id=4, base_render_bchw=base_render)
    second = content_loss._base_tokens(view_id=4, base_render_bchw=base_render)
    third = content_loss._base_tokens(view_id=7, base_render_bchw=base_render)

    assert first is second
    assert third is not first
    assert len(calls) == 2


def test_dino_losses_do_not_share_base_render_state(monkeypatch) -> None:
    first = _content_loss()
    second = _content_loss()
    calls = 0

    def fake_tokens(image_bchw: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return image_bchw.new_zeros((1, 2, 3))

    monkeypatch.setattr(first, "_patch_tokens", fake_tokens)
    monkeypatch.setattr(second, "_patch_tokens", fake_tokens)
    base_render = torch.zeros((1, 3, 28, 28))

    first._base_tokens(view_id=4, base_render_bchw=base_render)
    second._base_tokens(view_id=4, base_render_bchw=base_render)

    assert calls == 2


def test_dino_loss_caches_base_before_building_render_graph(monkeypatch) -> None:
    content_loss = _content_loss()
    events: list[str] = []

    def fake_base_tokens(
        *,
        view_id: int,
        base_render_bchw: torch.Tensor,
    ) -> torch.Tensor:
        assert view_id == 4
        events.append("base")
        return base_render_bchw.new_zeros((1, 2, 3))

    def fake_patch_tokens(image_bchw: torch.Tensor) -> torch.Tensor:
        events.append("render")
        return image_bchw.new_zeros((1, 2, 3))

    monkeypatch.setattr(content_loss, "_base_tokens", fake_base_tokens)
    monkeypatch.setattr(content_loss, "_patch_tokens", fake_patch_tokens)
    content_loss.loss(
        torch.zeros((1, 3, 28, 28)),
        torch.zeros((1, 3, 28, 28)),
        view_id=4,
    )

    assert events == ["base", "render"]


def test_dino_normalization_tensors_are_reused_by_dtype() -> None:
    content_loss = _content_loss()

    first = content_loss._normalization_for(torch.float32)
    second = content_loss._normalization_for(torch.float32)
    other_dtype = content_loss._normalization_for(torch.float64)

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert other_dtype[0].dtype == torch.float64
    assert other_dtype[0] is not first[0]


def test_dino_model_is_released_with_its_loss(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=FakeAutoModel),
    )
    content_loss = _content_loss()
    model = content_loss._load_model()
    model_reference = weakref.ref(model)

    del model, content_loss
    gc.collect()

    assert model_reference() is None
