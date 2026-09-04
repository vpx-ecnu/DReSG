from __future__ import annotations

import math
from pathlib import Path

from dresg.config import DReSGConfig
from dresg.models.gs.fitting import APPEARANCE_UPDATE_RULES

_HYDRA_METADATA_FILES = frozenset({"config.yaml", "hydra.yaml", "overrides.yaml"})


def _require_path(value: Path, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a pathlib.Path")
    return value


def _require_existing_path(path: Path, field_name: str, *, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise ValueError(f"{field_name} does not exist or is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise ValueError(f"{field_name} does not exist or is not a directory: {path}")


def _require_fresh_output_directory(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"data.output_dir exists and is not a directory: {path}")
    entries = list(path.iterdir())
    if not entries:
        return
    hydra_dir = path / ".hydra"
    if entries != [hydra_dir] or hydra_dir.is_symlink() or not hydra_dir.is_dir():
        raise FileExistsError(
            f"Training output directory contains stale artifacts: {path}"
        )
    metadata = list(hydra_dir.iterdir())
    names = {entry.name for entry in metadata}
    if names != _HYDRA_METADATA_FILES or any(
        entry.is_symlink() or not entry.is_file() for entry in metadata
    ):
        raise FileExistsError(
            f"Training output directory has invalid Hydra metadata: {hydra_dir}"
        )


def _require_positive(value: float, field_name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")


def _require_nonnegative(value: float, field_name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")


def _require_positive_integer(value: int, field_name: str) -> None:
    _require_integer(value, field_name)
    if value < 1:
        raise ValueError(f"{field_name} must be positive")


def _require_nonnegative_integer(value: int, field_name: str) -> None:
    _require_integer(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def validate_run_config(config: DReSGConfig, *, check_paths: bool = True) -> None:
    """Validate cross-field and filesystem invariants before loading models."""
    seed = config.runtime.seed
    _require_integer(seed, "runtime.seed")
    if seed < 0 or seed >= 2**32:
        raise ValueError("runtime.seed must be in the range 0..2^32-1")

    data = config.data
    base_ply = _require_path(data.base_ply, "data.base_ply")
    if base_ply.suffix != ".ply":
        raise ValueError("data.base_ply must use the .ply suffix")
    scene_dir = _require_path(data.scene_dir, "data.scene_dir")
    style_image = _require_path(data.style_image, "data.style_image")
    output_path = _require_path(data.output_dir, "data.output_dir")
    if check_paths:
        _require_existing_path(base_ply, "data.base_ply", kind="file")
        _require_existing_path(scene_dir, "data.scene_dir", kind="directory")
        _require_existing_path(style_image, "data.style_image", kind="file")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"data.output_dir exists and is not a directory: {output_path}")
    if check_paths:
        _require_fresh_output_directory(output_path)

    if not data.views:
        raise ValueError("data.views must contain at least one view")
    for view in data.views:
        _require_nonnegative_integer(view, "data.views entries")
    if len(set(data.views)) != len(data.views):
        raise ValueError("data.views must contain unique indices")
    _require_positive_integer(data.factor, "data.factor")

    rendering = config.rendering
    _require_positive(rendering.render_scale, "rendering.render_scale")
    _require_positive_integer(rendering.teacher_width, "rendering.teacher_width")
    _require_positive_integer(rendering.teacher_height, "rendering.teacher_height")
    if rendering.teacher_width % 64 or rendering.teacher_height % 64:
        raise ValueError("rendering teacher dimensions must be multiples of 64")

    optimization = config.guidance.optimization
    _require_positive_integer(
        optimization.num_inference_steps,
        "guidance.optimization.num_inference_steps",
    )
    _require_positive(optimization.learning_rate, "guidance.optimization.learning_rate")
    _require_positive_integer(
        optimization.inner_iterations,
        "guidance.optimization.inner_iterations",
    )
    _require_positive_integer(
        optimization.view_batch_size,
        "guidance.optimization.view_batch_size",
    )

    schedule = config.schedule
    if not schedule.prefixes:
        raise ValueError("schedule.prefixes must contain at least one prefix")
    for prefix in schedule.prefixes:
        _require_positive_integer(prefix, "schedule.prefixes entries")
    if schedule.prefixes != sorted(set(schedule.prefixes)):
        raise ValueError("schedule.prefixes must be unique and increasing")
    if max(schedule.prefixes) > optimization.num_inference_steps:
        raise ValueError(
            "Largest schedule prefix exceeds guidance.optimization.num_inference_steps"
        )
    _require_nonnegative_integer(schedule.fit_steps, "schedule.fit_steps")
    if schedule.max_stages is not None:
        _require_positive_integer(schedule.max_stages, "schedule.max_stages")
        if schedule.max_stages > len(schedule.prefixes):
            raise ValueError("schedule.max_stages must not exceed the prefix count")

    teacher = config.teacher
    if teacher.mode not in {
        "constant",
        "snr_balanced",
        "snr_triangle",
        "timestep_cosine",
    }:
        raise ValueError(
            "teacher.mode must be one of: constant, snr_balanced, "
            "snr_triangle, timestep_cosine"
        )
    _require_positive(teacher.scale, "teacher.scale")
    _require_positive(teacher.gamma_max, "teacher.gamma_max")

    appearance = config.appearance_optim
    _require_positive(appearance.lr, "appearance_optim.lr")
    _require_positive(appearance.geometry_lr, "appearance_optim.geometry_lr")
    if appearance.geometry_quat_lr is not None:
        _require_positive(appearance.geometry_quat_lr, "appearance_optim.geometry_quat_lr")
    for field_name in ("max_mean_delta", "max_scale_delta", "max_quat_delta"):
        _require_positive(getattr(appearance, field_name), f"appearance_optim.{field_name}")
    if appearance.optimize_geometry_quats and not appearance.optimize_geometry:
        raise ValueError(
            "appearance_optim.optimize_geometry_quats requires optimize_geometry"
        )

    image_loss = config.image_loss
    for field_name in (
        "lambda_l1",
        "lambda_dssim",
        "lambda_img_tv",
        "lambda_content3d",
    ):
        _require_nonnegative(getattr(image_loss, field_name), f"image_loss.{field_name}")
    _require_positive_integer(
        image_loss.content3d_dino_size,
        "image_loss.content3d_dino_size",
    )
    _require_nonnegative_integer(
        config.color_transfer.post_fit_steps,
        "color_transfer.post_fit_steps",
    )
    if config.appearance_update.rule not in APPEARANCE_UPDATE_RULES:
        raise ValueError(f"Unsupported appearance_update.rule: {config.appearance_update.rule}")
    if config.guidance.feedback.mode not in {"none", "render_latent"}:
        raise ValueError("guidance.feedback.mode must be one of: none, render_latent")

    backbone = config.guidance.backbone
    if not backbone.model_id.strip():
        raise ValueError("guidance.backbone.model_id must not be empty")
    if backbone.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "guidance.backbone.mixed_precision must be one of: no, fp16, bf16"
        )
    attention = config.guidance.attention
    if not attention.layers:
        raise ValueError("guidance.attention.layers must not be empty")
    if len(set(attention.layers)) != len(attention.layers):
        raise ValueError("guidance.attention.layers must contain unique names")
    if any(not layer.endswith(".attn1") for layer in attention.layers):
        raise ValueError("guidance.attention.layers must select self-attention .attn1 modules")
    _require_nonnegative(attention.content_weight, "guidance.attention.content_weight")
    _require_positive(attention.query_scale, "guidance.attention.query_scale")

    video = config.artifacts.video
    _require_positive_integer(video.fps, "artifacts.video.fps")
    _require_positive_integer(video.batch_size, "artifacts.video.batch_size")
    if video.path is not None:
        if not isinstance(video.path, Path):
            raise TypeError("artifacts.video.path must be a pathlib.Path or None")
        if check_paths:
            _require_existing_path(video.path, "artifacts.video.path", kind="file")
