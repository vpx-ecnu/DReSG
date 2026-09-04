"""Typed schema for values composed from the repository Hydra YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class DataConfig:
    base_ply: Path = MISSING
    scene_dir: Path = MISSING
    style_image: Path = MISSING
    output_dir: Path = MISSING
    views: list[int] = MISSING
    factor: int = MISSING


@dataclass
class RuntimeConfig:
    device: str = MISSING
    seed: int = MISSING
    offline_models: bool = MISSING


@dataclass
class RenderingConfig:
    render_scale: float = MISSING
    teacher_width: int = MISSING
    teacher_height: int = MISSING


@dataclass
class ViewSelectionConfig:
    dataset: str = MISSING
    scene: str = MISSING
    candidate_count: int = MISSING
    selected_count: int = MISSING
    coverage_ratio: float = MISSING


@dataclass
class ScheduleConfig:
    prefixes: list[int] = MISSING
    fit_steps: int = MISSING
    max_stages: int | None = MISSING

    @property
    def active_prefixes(self) -> tuple[int, ...]:
        if self.max_stages is None:
            return tuple(self.prefixes)
        return tuple(self.prefixes[: self.max_stages])


@dataclass
class TeacherConfig:
    mode: str = MISSING
    gamma_max: float = MISSING
    scale: float = MISSING


@dataclass
class AppearanceOptimizationConfig:
    lr: float = MISSING
    optimize_geometry: bool = MISSING
    optimize_geometry_quats: bool = MISSING
    geometry_lr: float = MISSING
    geometry_quat_lr: float | None = MISSING
    max_mean_delta: float = MISSING
    max_scale_delta: float = MISSING
    max_quat_delta: float = MISSING


@dataclass
class ImageLossConfig:
    lambda_l1: float = MISSING
    l1_use_unclamped_render: bool = MISSING
    lambda_dssim: float = MISSING
    lambda_img_tv: float = MISSING
    lambda_content3d: float = MISSING
    content3d_dino_model: str = MISSING
    content3d_dino_size: int = MISSING


@dataclass
class AppearanceUpdateConfig:
    rule: str = MISSING


@dataclass
class ColorTransferConfig:
    post_enabled: bool = MISSING
    post_fit_steps: int = MISSING


@dataclass
class DebugConfig:
    collect_stage_diagnostics: bool = MISSING


@dataclass
class GuidanceBackboneConfig:
    model_id: str = MISSING
    mixed_precision: str = MISSING


@dataclass
class GuidanceAttentionConfig:
    layers: list[str] = MISSING
    reference_add_noise: bool = MISSING
    content_weight: float = MISSING
    query_scale: float = MISSING


@dataclass
class GuidanceOptimizationConfig:
    num_inference_steps: int = MISSING
    learning_rate: float = MISSING
    inner_iterations: int = MISSING
    view_batch_size: int = MISSING


@dataclass
class GuidanceFeedbackConfig:
    mode: str = MISSING


@dataclass
class GuidanceConfig:
    backbone: GuidanceBackboneConfig = field(default_factory=GuidanceBackboneConfig)
    attention: GuidanceAttentionConfig = field(default_factory=GuidanceAttentionConfig)
    optimization: GuidanceOptimizationConfig = field(default_factory=GuidanceOptimizationConfig)
    feedback: GuidanceFeedbackConfig = field(default_factory=GuidanceFeedbackConfig)


@dataclass
class VideoConfig:
    path: Path | None = MISSING
    fps: int = MISSING
    batch_size: int = MISSING


@dataclass
class ArtifactsConfig:
    save_train_views: bool = MISSING
    save_style_image: bool = MISSING
    video: VideoConfig = field(default_factory=VideoConfig)


@dataclass
class DReSGConfig:
    """Complete application configuration after Hydra composition."""

    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    view_selection: ViewSelectionConfig = field(default_factory=ViewSelectionConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    appearance_optim: AppearanceOptimizationConfig = field(
        default_factory=AppearanceOptimizationConfig
    )
    image_loss: ImageLossConfig = field(default_factory=ImageLossConfig)
    appearance_update: AppearanceUpdateConfig = field(default_factory=AppearanceUpdateConfig)
    color_transfer: ColorTransferConfig = field(default_factory=ColorTransferConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)


_REGISTERED = False


def register_dresg_config() -> None:
    """Register the root schema once while leaving all values in YAML."""
    global _REGISTERED
    if _REGISTERED:
        return
    ConfigStore.instance().store(name="dresg_schema", node=DReSGConfig)
    _REGISTERED = True


def to_typed_config(config: DictConfig) -> DReSGConfig:
    """Resolve a composed Hydra config into real nested dataclass instances."""
    resolved = OmegaConf.to_object(config)
    if not isinstance(resolved, DReSGConfig):
        raise TypeError("Hydra config was not composed against DReSGConfig")
    return resolved
