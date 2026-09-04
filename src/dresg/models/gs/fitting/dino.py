"""Run-owned DINO patch content loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


def _dino_pixel_values(
    image_bchw: torch.Tensor,
    *,
    size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    image_bchw = image_bchw.clamp(0.0, 1.0)
    image_bchw = F.interpolate(
        image_bchw,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    return (image_bchw - mean) / std


def _cosine_patch_loss(
    render_tokens: torch.Tensor,
    base_tokens: torch.Tensor,
) -> torch.Tensor:
    if render_tokens.shape != base_tokens.shape:
        raise ValueError(
            "Render/base DINO token shapes must match: "
            f"render={tuple(render_tokens.shape)} base={tuple(base_tokens.shape)}"
        )
    render_normalized = F.normalize(render_tokens, dim=-1)
    base_normalized = F.normalize(base_tokens, dim=-1)
    return (1.0 - (render_normalized * base_normalized).sum(dim=-1)).mean()


class DinoPatchContentLoss:
    """Own one run's frozen DINO model and per-view base-render token cache."""

    def __init__(
        self,
        *,
        model_name: str,
        size: int,
        local_files_only: bool,
        device: torch.device,
    ) -> None:
        self._model_name = model_name
        self._size = size
        self._local_files_only = local_files_only
        self._device = torch.empty(0, device=device).device
        self._model: torch.nn.Module | None = None
        self._normalization_by_dtype: dict[
            torch.dtype,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._base_tokens_by_view: dict[int, torch.Tensor] = {}

    def _load_model(self) -> torch.nn.Module:
        if self._model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                self._model_name,
                local_files_only=self._local_files_only,
            )
            model.requires_grad_(False)
            self._model = model.to(self._device).eval()
        return self._model

    def _normalization_for(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalization = self._normalization_by_dtype.get(dtype)
        if normalization is None:
            mean = torch.tensor(
                _DINO_MEAN,
                device=self._device,
                dtype=dtype,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                _DINO_STD,
                device=self._device,
                dtype=dtype,
            ).view(1, 3, 1, 1)
            normalization = (mean, std)
            self._normalization_by_dtype[dtype] = normalization
        return normalization

    def _patch_tokens(self, image_bchw: torch.Tensor) -> torch.Tensor:
        model = self._load_model()
        model_dtype = next(model.parameters()).dtype
        mean, std = self._normalization_for(image_bchw.dtype)
        pixel_values = _dino_pixel_values(
            image_bchw,
            size=self._size,
            mean=mean,
            std=std,
        ).to(dtype=model_dtype)
        outputs = model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 1:].float()

    def _base_tokens(
        self,
        *,
        view_id: int,
        base_render_bchw: torch.Tensor,
    ) -> torch.Tensor:
        cached = self._base_tokens_by_view.get(view_id)
        if cached is None:
            with torch.no_grad():
                cached = self._patch_tokens(base_render_bchw.detach()).detach()
            self._base_tokens_by_view[view_id] = cached
        return cached

    def loss(
        self,
        render_bchw: torch.Tensor,
        base_render_bchw: torch.Tensor,
        *,
        view_id: int,
    ) -> torch.Tensor:
        base_tokens = self._base_tokens(
            view_id=view_id,
            base_render_bchw=base_render_bchw,
        )
        render_tokens = self._patch_tokens(render_bchw)
        return _cosine_patch_loss(render_tokens, base_tokens)
