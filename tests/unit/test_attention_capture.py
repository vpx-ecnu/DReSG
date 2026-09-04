from __future__ import annotations

import gc
import weakref

import pytest
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from dresg.models.diffusion.attention.capture import SelfAttentionExtractor


class _AttentionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = Attention(
            query_dim=8,
            cross_attention_dim=8,
            heads=2,
            dim_head=4,
            dropout=0.0,
        )
        self.attn2 = Attention(
            query_dim=8,
            cross_attention_dim=8,
            heads=2,
            dim_head=4,
            dropout=0.0,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attn1(hidden_states)


class _AttentionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _AttentionBlock()

    def forward(
        self,
        latent: torch.Tensor,
        _timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _ = encoder_hidden_states
        return self.block(latent)


def _extractor(model: torch.nn.Module) -> SelfAttentionExtractor:
    return SelfAttentionExtractor(
        unet=model,
        layer_names=("block.attn1",),
    )


def _extract(
    extractor: SelfAttentionExtractor,
    hidden_states: torch.Tensor,
):
    return extractor.extract(
        hidden_states,
        torch.tensor(10),
        torch.zeros(hidden_states.shape[0], 1, 8),
    )


def test_attention_capture_preserves_forward_and_input_gradient() -> None:
    torch.manual_seed(7)
    model = _AttentionModel()
    baseline_input = torch.randn(2, 5, 8, requires_grad=True)
    baseline_output = model(baseline_input, torch.tensor(10), torch.zeros(2, 1, 8))
    baseline_output.square().sum().backward()
    baseline_gradient = baseline_input.grad.detach().clone()

    _extractor(model)
    hooked_input = baseline_input.detach().clone().requires_grad_(True)
    hooked_output = model(hooked_input, torch.tensor(10), torch.zeros(2, 1, 8))
    hooked_output.square().sum().backward()

    torch.testing.assert_close(hooked_output, baseline_output)
    torch.testing.assert_close(hooked_input.grad, baseline_gradient)


def test_attention_capture_returns_named_atomic_qkv_output() -> None:
    torch.manual_seed(11)
    model = _AttentionModel()
    extractor = _extractor(model)
    hidden_states = torch.randn(2, 5, 8)

    features = _extract(extractor, hidden_states)

    assert features.layer_names == ("block.attn1",)
    assert len(features.queries) == len(features.keys) == len(features.values) == 1
    assert features.queries[0].shape == (2, 2, 5, 4)
    assert features.queries[0].shape == features.outputs[0].shape
    assert features.keys[0].shape == features.values[0].shape


def test_extractor_does_not_retain_returned_feature_tensors() -> None:
    model = _AttentionModel()
    extractor = _extractor(model)
    features = _extract(extractor, torch.randn(1, 4, 8))
    query_reference = weakref.ref(features.queries[0])

    del features
    gc.collect()

    assert query_reference() is None


def test_attention_capture_backpropagates_guidance_loss_to_input() -> None:
    model = _AttentionModel()
    extractor = _extractor(model)
    hidden_states = torch.randn(2, 5, 8, requires_grad=True)

    features = _extract(extractor, hidden_states)
    target = torch.nn.functional.scaled_dot_product_attention(
        features.queries[0],
        features.keys[0].flip(2),
        features.values[0].flip(2),
    ).detach()
    loss = torch.nn.functional.l1_loss(features.outputs[0], target)
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad.abs().sum() > 0


def test_attention_capture_registers_only_selected_layers() -> None:
    model = _AttentionModel()
    _extractor(model)

    assert len(model.block.attn1.to_q._forward_hooks) == 1
    assert len(model.block.attn1.to_k._forward_hooks) == 1
    assert len(model.block.attn1.to_v._forward_hooks) == 1
    assert len(model.block.attn1.to_out[0]._forward_pre_hooks) == 1


def test_registered_hooks_do_not_retain_discarded_extractor() -> None:
    model = _AttentionModel()
    extractor = _extractor(model)
    extractor_reference = weakref.ref(extractor)

    del extractor

    assert extractor_reference() is None
    model(torch.randn(1, 4, 8), torch.tensor(10), torch.zeros(1, 1, 8))


def test_attention_capture_is_collected_with_its_unet() -> None:
    model = _AttentionModel()
    extractor = _extractor(model)
    model_reference = weakref.ref(model)
    extractor_reference = weakref.ref(extractor)

    del extractor
    del model
    gc.collect()

    assert model_reference() is None
    assert extractor_reference() is None


def test_attention_capture_validates_every_layer_before_registering_hooks() -> None:
    model = _AttentionModel()

    with pytest.raises(ValueError, match="does not exist"):
        SelfAttentionExtractor(
            unet=model,
            layer_names=("block.attn1", "missing.attn1"),
        )

    assert not model.block.attn1.to_q._forward_hooks
    assert not model.block.attn1.to_k._forward_hooks
    assert not model.block.attn1.to_v._forward_hooks
    assert not model.block.attn1.to_out[0]._forward_pre_hooks


def test_attention_capture_rolls_back_partial_hook_registration(monkeypatch) -> None:
    model = _AttentionModel()

    def fail_registration(_hook):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(model.block.attn1.to_k, "register_forward_hook", fail_registration)
    with pytest.raises(RuntimeError, match="registration failed"):
        _extractor(model)

    assert not model.block.attn1.to_q._forward_hooks
    assert not model.block.attn1.to_k._forward_hooks
    assert not model.block.attn1.to_v._forward_hooks
    assert not model.block.attn1.to_out[0]._forward_pre_hooks


def test_attention_capture_rejects_cross_attention_name() -> None:
    model = _AttentionModel()

    with pytest.raises(ValueError, match="self-attention"):
        SelfAttentionExtractor(
            unet=model,
            layer_names=("block.attn2",),
        )


def test_attention_capture_rejects_unsupported_processor() -> None:
    class UnsupportedProcessor:
        def __call__(self, *_args, **_kwargs):
            raise NotImplementedError

    model = _AttentionModel()
    model.block.attn1.set_processor(UnsupportedProcessor())

    with pytest.raises(TypeError, match="AttnProcessor2_0"):
        _extractor(model)


def test_attention_capture_recovers_after_failed_forward() -> None:
    class RaisingProcessor:
        def __call__(self, attention, hidden_states, *_args, **_kwargs):
            attention.to_q(hidden_states)
            raise RuntimeError("expected failure")

    model = _AttentionModel()
    extractor = _extractor(model)
    model.block.attn1.set_processor(RaisingProcessor())
    with pytest.raises(RuntimeError, match="expected failure"):
        _extract(extractor, torch.randn(1, 4, 8))

    model.block.attn1.set_processor(AttnProcessor2_0())
    features = _extract(extractor, torch.randn(1, 4, 8))
    assert len(features.queries) == 1


def test_attention_capture_preserves_tiny_unet_output_and_gradient() -> None:
    torch.manual_seed(3)
    model = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=4,
    ).eval()
    layer_names = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, Attention) and name.endswith(".attn1")
    )
    sample = torch.randn(1, 4, 8, 8, requires_grad=True)
    timestep = torch.tensor([10])
    context = torch.randn(1, 5, 16)
    baseline = model(sample, timestep, context).sample
    baseline.square().mean().backward()
    baseline_gradient = sample.grad.detach().clone()

    extractor = SelfAttentionExtractor(
        unet=model,
        layer_names=layer_names,
    )
    hooked_sample = sample.detach().clone().requires_grad_(True)
    features = extractor.extract(hooked_sample, timestep, context)
    hooked = model(hooked_sample, timestep, context).sample
    hooked.square().mean().backward()

    torch.testing.assert_close(hooked, baseline)
    torch.testing.assert_close(hooked_sample.grad, baseline_gradient)
    assert features.layer_names == layer_names
    assert len(features.queries) == len(layer_names) == 4
