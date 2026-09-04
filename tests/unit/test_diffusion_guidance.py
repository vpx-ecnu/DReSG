from __future__ import annotations

import torch
import torch.nn.functional as F

from dresg.config import (
    GuidanceAttentionConfig,
    GuidanceFeedbackConfig,
    TeacherConfig,
)
from dresg.models.diffusion import DiffusionGuidance
from dresg.models.diffusion.attention.features import AttentionFeatures


class _Scheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1])
        self.alphas_cumprod = torch.tensor([0.9, 0.5])
        self.noise_calls = 0

    def add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        _timestep: torch.Tensor,
    ) -> torch.Tensor:
        self.noise_calls += 1
        return latent + noise


class _Codec:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return images.mean(dim=1, keepdim=True)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.repeat(1, 3, 1, 1)


class _Extractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self,
        latents: torch.Tensor,
        _timestep: torch.Tensor,
        _embeddings: torch.Tensor,
    ) -> AttentionFeatures:
        self.calls += 1
        batch = int(latents.shape[0])
        query = latents.mean(dim=(1, 2, 3)).reshape(batch, 1, 1, 1)
        key = query + 1.0
        value = query + 2.0
        output = F.scaled_dot_product_attention(query, key, value)
        return AttentionFeatures(
            layer_names=("up.attn1",),
            queries=(query,),
            keys=(key,),
            values=(value,),
            outputs=(output,),
        )


def _guidance(
    *,
    feedback_mode: str = "render_latent",
    reference_add_noise: bool = False,
    scheduler: _Scheduler | None = None,
    codec: _Codec | None = None,
    extractor: _Extractor | None = None,
    view_ids: tuple[int, ...] = (4, 7),
    image_height: int = 2,
    image_width: int = 2,
) -> DiffusionGuidance:
    scheduler = _Scheduler() if scheduler is None else scheduler
    codec = _Codec() if codec is None else codec
    extractor = _Extractor() if extractor is None else extractor
    base_renders = {view_id: torch.full((3, 2, 2), 0.2 + 0.05 * index) for index, view_id in enumerate(view_ids)}
    source_images = {view_id: torch.full((3, 2, 2), 0.4 + 0.05 * index) for index, view_id in enumerate(view_ids)}
    return DiffusionGuidance(
        scheduler=scheduler,
        codec=codec,
        attention_extractor=extractor,
        null_embeddings=torch.ones(1, 2, 3),
        image_height=image_height,
        image_width=image_width,
        style_image=torch.full((3, 2, 2), 0.6),
        base_renders_by_view=base_renders,
        source_images_by_view=source_images,
        teacher_config=TeacherConfig(
            mode="constant",
            gamma_max=1.25,
            scale=1.25,
        ),
        active_prefixes=(1,),
        attention=GuidanceAttentionConfig(
            layers=["up.attn1"],
            reference_add_noise=reference_add_noise,
            content_weight=0.1,
            query_scale=1.0,
        ),
        feedback=GuidanceFeedbackConfig(mode=feedback_mode),
    )



def test_diffusion_guidance_prepares_model_state_and_losses() -> None:
    extractor = _Extractor()
    guidance = _guidance(extractor=extractor)

    timestep = guidance.prepare_timestep(torch.tensor(1))
    batch = guidance.prepare_batch(timestep, (4, 7))
    losses = guidance.batch_losses(batch)

    assert guidance.view_ids == (4, 7)
    assert guidance.scale_at(1) == 1.25
    assert isinstance(timestep, DiffusionGuidance.TimestepState)
    assert isinstance(batch, DiffusionGuidance.OptimizationBatch)
    assert batch.view_ids == (4, 7)
    assert batch.latents.requires_grad
    assert losses.total >= 0.0
    assert extractor.calls == 3


def test_base_renders_and_source_images_are_encoded_separately() -> None:
    codec = _Codec()

    _guidance(codec=codec)

    assert codec.encode_calls == 5  # one style plus two images from each domain


def test_reference_noise_applies_only_to_style_and_content_features() -> None:
    scheduler = _Scheduler()
    guidance = _guidance(
        scheduler=scheduler,
        reference_add_noise=True,
    )

    timestep = guidance.prepare_timestep(torch.tensor(1))
    batch = guidance.prepare_batch(timestep, guidance.view_ids)
    guidance.batch_losses(batch)

    assert scheduler.noise_calls == 2


def test_guidance_preserves_requested_batch_order_and_commits_latents() -> None:
    guidance = _guidance(view_ids=(4, 7, 9))
    timestep = guidance.prepare_timestep(torch.tensor(1))
    batch = guidance.prepare_batch(timestep, (9, 4))

    with torch.no_grad():
        batch.latents.add_(0.1)
    guidance.commit_batch(batch)

    source = torch.zeros(3, 2, 2)
    view9 = guidance.teacher_image(view_id=9, source_rgb=source, scale=1.0)
    view4 = guidance.teacher_image(view_id=4, source_rgb=source, scale=1.0)
    torch.testing.assert_close(view9, torch.full_like(view9, 0.4))
    torch.testing.assert_close(view4, torch.full_like(view4, 0.3))


def test_diffusion_guidance_constructs_scaled_teacher_images() -> None:
    guidance = _guidance()
    source = torch.full((3, 4, 5), 0.1)

    teacher = guidance.teacher_image(
        view_id=4,
        source_rgb=source,
        scale=2.0,
    )

    expected = torch.sigmoid(
        torch.logit(source) + 2.0 * (torch.logit(torch.full_like(source, 0.2)) - torch.logit(source))
    )
    assert teacher.shape == source.shape
    torch.testing.assert_close(teacher, expected)


def test_diffusion_guidance_applies_render_latent_feedback() -> None:
    guidance = _guidance(feedback_mode="render_latent")
    render = torch.full((3, 2, 2), 0.7)

    projection_gap = guidance.project_render(
        view_id=4,
        render_rgb=render,
    )
    teacher = guidance.teacher_image(view_id=4, source_rgb=render, scale=1.0)

    torch.testing.assert_close(projection_gap, torch.tensor(0.5))
    torch.testing.assert_close(teacher, render)


def test_diffusion_guidance_reports_projection_without_feedback_writeback() -> None:
    guidance = _guidance(feedback_mode="none")
    render = torch.full((3, 2, 2), 0.7)

    projection_gap = guidance.project_render(view_id=4, render_rgb=render)
    teacher = guidance.teacher_image(view_id=4, source_rgb=render, scale=1.0)

    torch.testing.assert_close(projection_gap, torch.tensor(0.5))
    torch.testing.assert_close(teacher, torch.full_like(render, 0.2))
