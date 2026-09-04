"""Run-specific diffusion guidance state and operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline

from dresg.config import (
    GuidanceAttentionConfig,
    GuidanceBackboneConfig,
    GuidanceFeedbackConfig,
    TeacherConfig,
)
from dresg.models.diffusion.attention.capture import SelfAttentionExtractor
from dresg.models.diffusion.attention.features import AttentionFeatures
from dresg.models.diffusion.attention.losses import (
    GuidanceLosses,
    attention_guidance_losses,
)
from dresg.models.diffusion.latents.bank import ViewLatentBank
from dresg.models.diffusion.latents.codec import LatentCodec
from dresg.models.diffusion.scheduling.scale import TeacherScaleSchedule
from dresg.utils.images import preprocess_rgb_tensor


def _weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "no":
        return torch.float32
    raise ValueError(f"Unsupported diffusion mixed precision: {mixed_precision}")


def _encode_view_images(
    *,
    codec: LatentCodec,
    images_by_view: Mapping[int, torch.Tensor],
    image_height: int,
    image_width: int,
) -> dict[int, torch.Tensor]:
    return {
        view_id: codec.encode(preprocess_rgb_tensor(image, image_height, image_width))
        for view_id, image in images_by_view.items()
    }


def _scale_teacher_residual_logit(
    *,
    source_rgb: torch.Tensor,
    teacher_rgb: torch.Tensor,
    scale: float,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    if abs(scale - 1.0) < 1.0e-8:
        return teacher_rgb
    source_logit = torch.logit(source_rgb.clamp(eps, 1.0 - eps))
    teacher_logit = torch.logit(teacher_rgb.clamp(eps, 1.0 - eps))
    return torch.sigmoid(source_logit + scale * (teacher_logit - source_logit))


class DiffusionGuidance:
    """Own per-run diffusion references, evolving latents, and feedback state."""

    @dataclass(frozen=True, slots=True)
    class TimestepState:
        """Reference features shared only within one guidance timestep."""

        timestep: torch.Tensor
        style_features: AttentionFeatures

    @dataclass(frozen=True, slots=True)
    class OptimizationBatch:
        """One trainable latent batch and its fixed guidance references."""

        view_ids: tuple[int, ...]
        latents: torch.Tensor
        timestep: DiffusionGuidance.TimestepState
        content_features: AttentionFeatures

    def __init__(
        self,
        *,
        scheduler: DDIMScheduler,
        codec: LatentCodec,
        attention_extractor: SelfAttentionExtractor,
        null_embeddings: torch.Tensor,
        image_height: int,
        image_width: int,
        style_image: torch.Tensor,
        base_renders_by_view: Mapping[int, torch.Tensor],
        source_images_by_view: Mapping[int, torch.Tensor],
        teacher_config: TeacherConfig,
        active_prefixes: Sequence[int],
        attention: GuidanceAttentionConfig,
        feedback: GuidanceFeedbackConfig,
    ) -> None:
        scale_schedule = TeacherScaleSchedule.from_timeline(
            teacher=teacher_config,
            active_prefixes=active_prefixes,
            timesteps=scheduler.timesteps,
            alphas_cumprod=scheduler.alphas_cumprod,
        )
        style_latent = codec.encode(preprocess_rgb_tensor(style_image, image_height, image_width))
        encoded_base_renders = _encode_view_images(
            codec=codec,
            images_by_view=base_renders_by_view,
            image_height=image_height,
            image_width=image_width,
        )
        current_latents = ViewLatentBank(encoded_base_renders)
        encoded_source_images = _encode_view_images(
            codec=codec,
            images_by_view=source_images_by_view,
            image_height=image_height,
            image_width=image_width,
        )
        content_latents = ViewLatentBank(encoded_source_images)
        if current_latents.view_ids != content_latents.view_ids:
            raise ValueError(
                "Base-render/source-image latent view order must match: "
                f"base={current_latents.view_ids} source={content_latents.view_ids}"
            )
        self._scheduler = scheduler
        self._codec = codec
        self._attention_extractor = attention_extractor
        self._null_embeddings = null_embeddings
        self._image_height = image_height
        self._image_width = image_width
        self._style_latent = style_latent
        self._current_latents = current_latents
        self._content_latents = content_latents
        self._scale_schedule = scale_schedule
        self._reference_add_noise = attention.reference_add_noise
        self._content_weight = attention.content_weight
        self._query_scale = attention.query_scale
        self._feedback_mode = feedback.mode

    def _extract_attention(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        *,
        add_noise: bool,
    ) -> AttentionFeatures:
        if latent.ndim != 4:
            raise ValueError("Attention latent input must have shape [B, C, H, W]")
        unet_input = latent
        if add_noise:
            unet_input = self._scheduler.add_noise(
                latent,
                torch.randn_like(latent),
                timestep,
            )
        embeddings = self._null_embeddings.expand(latent.shape[0], -1, -1)
        return self._attention_extractor.extract(
            unet_input,
            timestep,
            embeddings,
        )

    def _extract_reference_attention(
        self,
        *,
        latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> AttentionFeatures:
        with torch.no_grad():
            return self._extract_attention(
                latent,
                timestep,
                add_noise=self._reference_add_noise,
            )

    def prepare_timestep(self, timestep: torch.Tensor) -> TimestepState:
        return self.TimestepState(
            timestep=timestep,
            style_features=self._extract_reference_attention(
                latent=self._style_latent,
                timestep=timestep,
            ),
        )

    def prepare_batch(
        self,
        timestep: TimestepState,
        view_ids: Sequence[int],
    ) -> OptimizationBatch:
        ids = tuple(view_ids)
        current_latents = self._current_latents.batch(ids)
        content_latents = self._content_latents.batch(ids)
        if current_latents.shape != content_latents.shape:
            raise ValueError(
                "Current/content latent batch shapes must match: "
                f"current={tuple(current_latents.shape)} content={tuple(content_latents.shape)}"
            )
        return self.OptimizationBatch(
            view_ids=ids,
            latents=current_latents.detach().float().requires_grad_(),
            timestep=timestep,
            content_features=self._extract_reference_attention(
                latent=content_latents,
                timestep=timestep.timestep,
            ),
        )

    def batch_losses(self, batch: OptimizationBatch) -> GuidanceLosses:
        current_features = self._extract_attention(
            batch.latents,
            batch.timestep.timestep,
            add_noise=False,
        )
        return attention_guidance_losses(
            current=current_features,
            style=batch.timestep.style_features,
            content=batch.content_features,
            content_weight=self._content_weight,
            query_scale=self._query_scale,
        )

    def commit_batch(self, batch: OptimizationBatch) -> None:
        self._current_latents.replace_batch(
            batch.view_ids,
            batch.latents.detach(),
        )

    @property
    def timesteps(self) -> torch.Tensor:
        return self._scheduler.timesteps

    @property
    def view_ids(self) -> tuple[int, ...]:
        return self._current_latents.view_ids

    def scale_at(self, prefix: int) -> float:
        return self._scale_schedule.scale_at(prefix)

    @torch.no_grad()
    def teacher_image(
        self,
        *,
        view_id: int,
        source_rgb: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if source_rgb.ndim != 3 or source_rgb.shape[0] != 3:
            raise ValueError("Teacher image source must have shape [3, H, W]")
        teacher_unscaled_rgb = self._codec.decode(self._current_latents[view_id])[0]
        teacher_unscaled_rgb = preprocess_rgb_tensor(
            teacher_unscaled_rgb,
            source_rgb.shape[-2],
            source_rgb.shape[-1],
        )[0]
        return _scale_teacher_residual_logit(
            source_rgb=source_rgb,
            teacher_rgb=teacher_unscaled_rgb,
            scale=scale,
        )

    @torch.no_grad()
    def project_render(
        self,
        *,
        view_id: int,
        render_rgb: torch.Tensor,
    ) -> torch.Tensor:
        render_teacher = preprocess_rgb_tensor(
            render_rgb,
            self._image_height,
            self._image_width,
        )
        projected_latent = self._codec.encode(render_teacher)
        current_latent = self._current_latents[view_id]
        projection_gap = F.l1_loss(projected_latent, current_latent)
        if self._feedback_mode == "render_latent":
            self._current_latents.replace_batch([view_id], projected_latent)
        return projection_gap


def build_diffusion_guidance(
    *,
    backbone: GuidanceBackboneConfig,
    image_height: int,
    image_width: int,
    style_image: torch.Tensor,
    base_renders_by_view: Mapping[int, torch.Tensor],
    source_images_by_view: Mapping[int, torch.Tensor],
    teacher_config: TeacherConfig,
    active_prefixes: Sequence[int],
    attention: GuidanceAttentionConfig,
    num_inference_steps: int,
    feedback: GuidanceFeedbackConfig,
    offline_models: bool,
    device: torch.device,
) -> DiffusionGuidance:
    """Build one run-owned diffusion model from explicit resources and inputs."""
    weight_dtype = _weight_dtype(backbone.mixed_precision)
    scheduler = DDIMScheduler.from_pretrained(
        backbone.model_id,
        subfolder="scheduler",
        local_files_only=offline_models,
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        backbone.model_id,
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=offline_models,
    )
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.enable_vae_slicing()
    pipeline.unet.to(device, dtype=weight_dtype)
    pipeline.vae.to(device, dtype=weight_dtype)
    pipeline.text_encoder.to(device, dtype=weight_dtype)
    null_embeddings = pipeline.encode_prompt("", device, 1, False)[0].detach()
    pipeline.scheduler.set_timesteps(num_inference_steps)
    codec = LatentCodec(
        vae=pipeline.vae,
        device=device,
        weight_dtype=weight_dtype,
    )
    attention_extractor = SelfAttentionExtractor(
        unet=pipeline.unet,
        layer_names=tuple(attention.layers),
    )
    return DiffusionGuidance(
        scheduler=pipeline.scheduler,
        codec=codec,
        attention_extractor=attention_extractor,
        null_embeddings=null_embeddings,
        image_height=image_height,
        image_width=image_width,
        style_image=style_image,
        base_renders_by_view=base_renders_by_view,
        source_images_by_view=source_images_by_view,
        teacher_config=teacher_config,
        active_prefixes=active_prefixes,
        attention=attention,
        feedback=feedback,
    )
