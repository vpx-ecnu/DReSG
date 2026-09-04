from __future__ import annotations

import torch

import dresg.models.diffusion.guidance as guidance_module
from dresg.config import (
    GuidanceAttentionConfig,
    GuidanceBackboneConfig,
    GuidanceFeedbackConfig,
    TeacherConfig,
)
from dresg.models.diffusion import (
    DiffusionGuidance,
    build_diffusion_guidance,
)


class _DeviceModule:
    def __init__(self) -> None:
        self.to_calls = []
        self.requires_grad_calls = []

    def requires_grad_(self, enabled: bool):
        self.requires_grad_calls.append(enabled)
        return self

    def to(self, device, **kwargs):
        self.to_calls.append((device, kwargs))
        return self


class _Scheduler:
    from_pretrained_calls = []

    def __init__(self) -> None:
        self.timesteps = None
        self.alphas_cumprod = torch.linspace(0.99, 0.01, 1000)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        subfolder: str,
        local_files_only: bool,
    ):
        cls.from_pretrained_calls.append((model_id, subfolder, local_files_only))
        return cls()

    def set_timesteps(self, steps: int) -> None:
        self.timesteps = torch.arange(steps)

    def add_noise(self, latent, noise, _timestep):
        return latent + noise


class _Pipeline:
    from_pretrained_calls = []
    last_instance = None

    def __init__(self, scheduler: _Scheduler) -> None:
        type(self).last_instance = self
        self.scheduler = scheduler
        self.unet = _DeviceModule()
        self.vae = _DeviceModule()
        self.text_encoder = _DeviceModule()
        self.vae_slicing_enabled = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        scheduler: _Scheduler,
        safety_checker,
        torch_dtype: torch.dtype,
        local_files_only: bool,
    ):
        cls.from_pretrained_calls.append((model_id, scheduler, safety_checker, torch_dtype, local_files_only))
        return cls(scheduler)

    def enable_vae_slicing(self) -> None:
        self.vae_slicing_enabled = True

    def encode_prompt(self, *_args):
        return (torch.ones(1, 2, 3),)


class _Codec:
    calls = []

    def __init__(self, *, vae, device, weight_dtype) -> None:
        self.calls.append((vae, device, weight_dtype))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=1, keepdim=True)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.repeat(1, 3, 1, 1)


class _Extractor:
    init_calls = []

    def __init__(self, *, unet, layer_names) -> None:
        self.init_calls.append((unet, layer_names))

    def extract(self, latent, timestep, embeddings):
        return (latent, timestep, embeddings)


def _load_guidance(monkeypatch) -> DiffusionGuidance:
    _Scheduler.from_pretrained_calls.clear()
    _Pipeline.from_pretrained_calls.clear()
    _Codec.calls.clear()
    _Extractor.init_calls.clear()
    monkeypatch.setattr(guidance_module, "DDIMScheduler", _Scheduler)
    monkeypatch.setattr(guidance_module, "StableDiffusionPipeline", _Pipeline)
    monkeypatch.setattr(guidance_module, "LatentCodec", _Codec)
    monkeypatch.setattr(guidance_module, "SelfAttentionExtractor", _Extractor)
    base_renders = {4: torch.full((3, 2, 2), 0.2)}
    source_images = {4: torch.full((3, 2, 2), 0.4)}
    return build_diffusion_guidance(
        backbone=GuidanceBackboneConfig(
            model_id="model",
            mixed_precision="bf16",
        ),
        image_height=2,
        image_width=2,
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
            reference_add_noise=False,
            content_weight=0.1,
            query_scale=1.0,
        ),
        num_inference_steps=7,
        feedback=GuidanceFeedbackConfig(mode="render_latent"),
        offline_models=True,
        device=torch.device("cpu"),
    )


def test_diffusion_guidance_loads_and_owns_upstream_components(monkeypatch) -> None:
    guidance = _load_guidance(monkeypatch)

    assert _Scheduler.from_pretrained_calls == [("model", "scheduler", True)]
    assert _Pipeline.from_pretrained_calls[0][0] == "model"
    assert _Pipeline.from_pretrained_calls[0][2] is None
    assert _Pipeline.from_pretrained_calls[0][3] is torch.bfloat16
    assert _Codec.calls[0][0] is _Pipeline.last_instance.vae
    assert _Extractor.init_calls == [(_Pipeline.last_instance.unet, ("up.attn1",))]
    assert guidance.timesteps.shape == (7,)


def test_diffusion_guidance_optionally_adds_noise(monkeypatch) -> None:
    guidance = _load_guidance(monkeypatch)
    latent = torch.zeros(1, 4, 2, 2)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    captured = guidance._extract_attention(
        latent,
        torch.tensor(10),
        add_noise=True,
    )

    torch.testing.assert_close(captured[0], torch.ones_like(latent))


def test_diffusion_guidance_expands_null_embeddings_for_latent_batch(
    monkeypatch,
) -> None:
    guidance = _load_guidance(monkeypatch)

    captured = guidance._extract_attention(
        torch.zeros(4, 1, 1, 1),
        torch.tensor(10),
        add_noise=False,
    )

    assert captured[2].shape == (4, 2, 3)
    assert captured[2].stride(0) == 0
