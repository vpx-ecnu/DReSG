"""CLIP and DINO feature extraction for paper metrics."""

from __future__ import annotations

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
        features = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(
                [self._preprocess(load_pil_rgb(path)) for path in batch_paths]
            ).to(self._device)
            encoded = self._model.encode_image(batch)
            encoded = F.normalize(encoded, p=2, dim=-1, eps=1e-8)
            features.append(encoded.float().cpu())
        return torch.cat(features, dim=0)


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
        return torch.cat(features, dim=0)


def mean_cosine_to_reference(features: torch.Tensor, reference: torch.Tensor) -> float:
    scores = features @ reference.mean(dim=0, keepdim=True).T
    return scores.squeeze(1).mean().item()


def mean_pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Pairwise feature shapes must match: a={tuple(a.shape)}, b={tuple(b.shape)}")
    return (a * b).sum(dim=-1).mean().item()
