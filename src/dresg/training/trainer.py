from __future__ import annotations

import torch
from tqdm import tqdm

from dresg.config import DReSGConfig
from dresg.data.cameras import Cameras, build_scaled_cameras
from dresg.data.colmap import ColmapScene, load_colmap_scene
from dresg.data.images import ViewImages, load_source_view_images
from dresg.inference.paths import load_video_path_for_scene
from dresg.models.diffusion import DiffusionGuidance, build_diffusion_guidance
from dresg.models.gs import GaussianScene, build_gaussian_scene
from dresg.models.gs.fitting import DinoPatchContentLoss
from dresg.training.optimization.gs import GaussianOptimizer
from dresg.training.output import build_training_progress
from dresg.training.stages import ColorStage, FeedbackStage, GuidanceStage
from dresg.training.validation import validate_run_config
from dresg.utils.images import load_rgb_image, save_rgb
from dresg.utils.seed import seed_random_generators


class DReSGTrainer:
    def __init__(self, config: DReSGConfig) -> None:
        validate_run_config(config)
        seed_random_generators(config.runtime.seed)
        self.config = config
        device = torch.device(config.runtime.device)
        config.data.output_dir.mkdir(parents=True, exist_ok=True)

        self.source: ColmapScene = load_colmap_scene(
            scene_dir=config.data.scene_dir,
            factor=config.data.factor,
        )
        self.cameras: Cameras = build_scaled_cameras(
            source=self.source,
            view_ids=config.data.views,
            device=device,
            render_scale=config.rendering.render_scale,
            label="active",
        )
        video_path = config.artifacts.video.path
        if video_path is not None:
            load_video_path_for_scene(video_path, self.source)

        appearance = config.appearance_optim
        self.scene: GaussianScene = build_gaussian_scene(
            ply_path=config.data.base_ply,
            device=device,
            optimize_geometry=appearance.optimize_geometry,
            optimize_quats=appearance.optimize_geometry_quats,
            max_mean_delta=appearance.max_mean_delta,
            max_scale_delta=appearance.max_scale_delta,
            max_quat_delta=appearance.max_quat_delta,
        )
        self.scene_optimizer = GaussianOptimizer(
            self.scene,
            self.cameras,
            appearance,
        )
        with torch.no_grad():
            base_renders = self.scene.render_current_images(self.cameras)
        self.base_renders: ViewImages = base_renders
        self.progress = build_training_progress(device, config.data.output_dir)

        style = load_rgb_image(config.data.style_image, device=device)
        if config.artifacts.save_style_image:
            save_rgb(config.data.output_dir / "style_image.png", style)
        source_images = load_source_view_images(self.source, self.cameras)

        self.guidance: DiffusionGuidance = build_diffusion_guidance(
            backbone=config.guidance.backbone,
            image_height=config.rendering.teacher_height,
            image_width=config.rendering.teacher_width,
            style_image=style,
            base_renders_by_view=self.base_renders,
            source_images_by_view=source_images,
            teacher_config=config.teacher,
            active_prefixes=config.schedule.active_prefixes,
            attention=config.guidance.attention,
            num_inference_steps=config.guidance.optimization.num_inference_steps,
            feedback=config.guidance.feedback,
            offline_models=config.runtime.offline_models,
            device=device,
        )

        image_loss = config.image_loss
        self.content_loss: DinoPatchContentLoss | None = None
        if image_loss.lambda_content3d > 0.0:
            self.content_loss = DinoPatchContentLoss(
                model_name=image_loss.content3d_dino_model,
                size=image_loss.content3d_dino_size,
                local_files_only=config.runtime.offline_models,
                device=device,
            )

    def run(self) -> None:
        self._run_guidance_stages()
        if self.config.color_transfer.post_enabled:
            self._run_post_color_stage()
        self.progress.finalize(
            scene=self.scene,
            source=self.source,
            cameras=self.cameras,
            artifacts=self.config.artifacts,
            rendering=self.config.rendering,
        )

    def _run_guidance_stages(self) -> None:
        guidance = self.guidance
        schedule = self.config.schedule
        guidance_stage = GuidanceStage(
            guidance,
            self.config.guidance.optimization,
            self.progress.runtime_metrics,
        )
        feedback_stage = FeedbackStage(
            self.scene,
            self.cameras,
            self.scene_optimizer,
            self.base_renders,
            guidance,
            self.progress.runtime_metrics,
            self.content_loss,
            image_loss=self.config.image_loss,
            appearance_update_rule=self.config.appearance_update.rule,
            collect_diagnostics=self.config.debug.collect_stage_diagnostics,
        )
        previous_prefix = 0
        with tqdm(
            total=schedule.active_prefixes[-1],
            desc="DReSG guidance",
            unit="step",
            dynamic_ncols=True,
            disable=None,
        ) as progress_bar:
            for prefix in schedule.active_prefixes:
                for timestep in guidance.timesteps[previous_prefix:prefix]:
                    metrics = guidance_stage.run(timestep)
                    latest = self.progress.record_guidance(prefix=prefix, metrics=metrics)
                    progress_bar.update()
                    progress_bar.set_postfix(
                        prefix=prefix,
                        style=f"{latest.style_loss:.4f}",
                        content=f"{latest.content_loss:.4f}",
                    )
                progress_bar.set_postfix_str(f"feedback prefix={prefix}")
                row = self.progress.record_stage(
                    feedback_stage.run(
                        prefix=prefix,
                        fit_steps=schedule.fit_steps,
                    )
                )
                progress_bar.set_postfix(
                    prefix=prefix,
                    teacher=f"{row['teacher_l1']:.4f}",
                    projection=f"{row['projection_gap_l1']:.4f}",
                )
                previous_prefix = prefix

    def _run_post_color_stage(self) -> None:
        config = self.config
        metrics = ColorStage(
            self.scene,
            self.cameras,
            self.scene_optimizer,
            style_image=config.data.style_image,
            image_loss=config.image_loss,
        ).run(fit_steps=config.color_transfer.post_fit_steps)
        self.progress.update_final_stage(metrics)
