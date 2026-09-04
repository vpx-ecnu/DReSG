"""Self-attention capture for Diffusers U-Nets."""

from __future__ import annotations

import weakref
from typing import Literal

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from torch.utils.hooks import RemovableHandle

from dresg.models.diffusion.attention.features import AttentionFeatures


def _split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    batch_size, token_count, inner_dim = tensor.shape
    if inner_dim % heads != 0:
        raise ValueError(f"Attention width {inner_dim} is not divisible by {heads} heads")
    return tensor.view(batch_size, token_count, heads, inner_dim // heads).transpose(1, 2)


AttentionField = Literal["query", "key", "value", "output"]
_ATTENTION_FIELDS: tuple[AttentionField, ...] = ("query", "key", "value", "output")


class SelfAttentionExtractor:
    """Capture named self-attention layers without changing their processors."""

    def __init__(
        self,
        *,
        unet: UNet2DConditionModel,
        layer_names: tuple[str, ...],
    ) -> None:
        if not layer_names:
            raise ValueError("At least one self-attention layer must be selected")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("Selected self-attention layer names must be unique")
        parameter = next(unet.parameters())
        self._unet = unet
        self._autocast_device_type = parameter.device.type
        self._autocast_dtype = parameter.dtype
        self._autocast_enabled = parameter.dtype in {torch.float16, torch.bfloat16}
        self._layer_names = layer_names
        self._captured: dict[tuple[str, AttentionField], torch.Tensor] = {}
        self._active = False
        self._register()

    def _register(self) -> None:
        modules = dict(self._unet.named_modules())
        selected: list[tuple[str, Attention]] = []
        for layer_name in self._layer_names:
            module = modules.get(layer_name)
            if not isinstance(module, Attention):
                raise ValueError(
                    f"Selected attention layer does not exist or is not Diffusers Attention: {layer_name}"
                )
            if not layer_name.endswith(".attn1"):
                raise ValueError(f"Selected layer must be self-attention (.attn1): {layer_name}")
            if not isinstance(module.processor, AttnProcessor2_0):
                raise TypeError(
                    "DReSG attention capture requires Diffusers AttnProcessor2_0 at "
                    f"{layer_name}; found {module.processor.__class__.__name__}"
                )
            selected.append((layer_name, module))

        handles: list[RemovableHandle] = []
        try:
            for layer_name, module in selected:
                handles.extend(self._register_layer(layer_name, module))
        except BaseException:
            for handle in handles:
                handle.remove()
            raise

    def _record(
        self,
        layer_name: str,
        field: AttentionField,
        tensor: torch.Tensor,
    ) -> None:
        key = (layer_name, field)
        if key in self._captured:
            raise RuntimeError(
                f"Attention layer {layer_name} produced {field} more than once in one U-Net pass"
            )
        self._captured[key] = tensor

    def _freeze(self) -> AttentionFeatures:
        missing = [
            f"{layer_name}:{field}"
            for layer_name in self._layer_names
            for field in _ATTENTION_FIELDS
            if (layer_name, field) not in self._captured
        ]
        if missing:
            raise RuntimeError(f"Incomplete self-attention capture: {missing}")

        def collect(field: AttentionField) -> tuple[torch.Tensor, ...]:
            return tuple(self._captured[(layer_name, field)] for layer_name in self._layer_names)

        return AttentionFeatures(
            layer_names=self._layer_names,
            queries=collect("query"),
            keys=collect("key"),
            values=collect("value"),
            outputs=collect("output"),
        )

    def _register_layer(
        self,
        layer_name: str,
        attention: Attention,
    ) -> list[RemovableHandle]:
        heads = int(attention.heads)
        extractor_ref = weakref.ref(self)

        def record_projection(field: AttentionField):
            def hook(
                _module: torch.nn.Module,
                _args: tuple[object, ...],
                output: torch.Tensor,
            ) -> None:
                extractor = extractor_ref()
                if extractor is not None and extractor._active:
                    extractor._record(layer_name, field, _split_heads(output, heads))

            return hook

        def record_output(
            _module: torch.nn.Module,
            args: tuple[object, ...],
        ) -> None:
            extractor = extractor_ref()
            if extractor is None or not extractor._active:
                return
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(
                    f"Attention output projection for {layer_name} received no tensor input"
                )
            extractor._record(layer_name, "output", _split_heads(args[0], heads))

        handles: list[RemovableHandle] = []
        try:
            handles.append(attention.to_q.register_forward_hook(record_projection("query")))
            handles.append(attention.to_k.register_forward_hook(record_projection("key")))
            handles.append(attention.to_v.register_forward_hook(record_projection("value")))
            handles.append(attention.to_out[0].register_forward_pre_hook(record_output))
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def extract(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> AttentionFeatures:
        if self._active:
            raise RuntimeError("SelfAttentionExtractor is not reentrant")
        self._captured.clear()
        self._active = True
        try:
            with torch.autocast(
                device_type=self._autocast_device_type,
                dtype=self._autocast_dtype,
                enabled=self._autocast_enabled,
            ):
                self._unet(latent, timestep, encoder_hidden_states=embeddings)
            features = self._freeze()
        except BaseException:
            self._active = False
            self._captured.clear()
            raise
        self._active = False
        self._captured.clear()
        return features
