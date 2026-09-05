"""CLIP and DINO feature extraction for paper metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL.Image import Image
from transformers import AutoImageProcessor, AutoModel, PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from dresg.utils.images import load_pil_rgb

DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_DINO_MODEL = "facebook/dinov2-base"


def _validate_request(paths: Sequence[Path], batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("Feature batch size must be an integer")
    if batch_size < 1:
        raise ValueError("Feature batch size must be positive")
    if not paths:
        raise ValueError("Feature extraction requires at least one image")


def _validated_features(features: list[torch.Tensor], *, name: str) -> torch.Tensor:
    encoded = torch.cat(features, dim=0)
    if encoded.ndim != 2 or not all(encoded.shape):
        raise ValueError(f"{name} features must be a non-empty [N, D] matrix")
    if not torch.isfinite(encoded).all():
        raise ValueError(f"{name} features must be finite")
    if (encoded.norm(dim=-1) == 0).any():
        raise ValueError(f"{name} features must have nonzero norms")
    return encoded


def _mean_cosine(scores: torch.Tensor) -> float:
    value = scores.mean().item()
    if not math.isfinite(value):
        raise ValueError("Cosine similarity must be finite and non-empty")
    return value


def _clip_model_source(model_name: str, offline_models: bool) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_file() or not offline_models:
        return str(model_path) if model_path.is_file() else model_name
    from clip.clip import _MODELS

    model_urls = _MODELS
    if model_name not in model_urls:
        raise FileNotFoundError(f"Offline CLIP model is not a local file: {model_name}")
    checkpoint = Path.home() / ".cache" / "clip" / model_urls[model_name].rsplit("/", 1)[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Offline CLIP checkpoint not found: {checkpoint}")
    return str(checkpoint)


class ClipEncoder:
    """Own one CLIP image model and its preprocessing transform."""

    def __init__(
        self,
        *,
        model_name: str,
        device: torch.device,
        offline_models: bool,
    ) -> None:
        import clip

        source = _clip_model_source(model_name, offline_models)
        model, preprocess = clip.load(source, device=device)
        self._model = model.eval()
        self._preprocess: Callable[[Image], torch.Tensor] = preprocess
        self._device = device

    @torch.no_grad()
    def encode(self, paths: Sequence[Path], *, batch_size: int) -> torch.Tensor:
        _validate_request(paths, batch_size)
        features = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(
                [self._preprocess(load_pil_rgb(path)) for path in batch_paths]
            ).to(self._device)
            encoded = self._model.encode_image(batch)
            encoded = F.normalize(encoded, p=2, dim=-1, eps=1e-8)
            features.append(encoded.float().cpu())
        return _validated_features(features, name="CLIP")


class DinoEncoder:
    """Own one DINO image model and its preprocessing transform."""

    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        offline_models: bool,
    ) -> None:
        self._processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=offline_models,
        )
        self._model: PreTrainedModel = AutoModel.from_pretrained(
            model_id,
            local_files_only=offline_models,
        ).to(device).eval()
        self._device = device

    @torch.no_grad()
    def encode(self, paths: Sequence[Path], *, batch_size: int) -> torch.Tensor:
        _validate_request(paths, batch_size)
        features = []
        for start in range(0, len(paths), batch_size):
            images = [
                load_pil_rgb(path) for path in paths[start : start + batch_size]
            ]
            inputs = self._processor(images=images, return_tensors="pt").to(
                self._device
            )
            outputs = self._model(**inputs)
            encoded = F.normalize(outputs.pooler_output, p=2, dim=-1, eps=1e-8)
            features.append(encoded.float().cpu())
        return _validated_features(features, name="DINO")


def mean_cosine_to_reference(features: torch.Tensor, reference: torch.Tensor) -> float:
    scores = features @ reference.mean(dim=0, keepdim=True).T
    return _mean_cosine(scores.squeeze(1))


def mean_pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Pairwise feature shapes must match: a={tuple(a.shape)}, b={tuple(b.shape)}")
    return _mean_cosine((a * b).sum(dim=-1))
