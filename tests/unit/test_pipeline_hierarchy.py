from __future__ import annotations

import importlib.util


def test_new_module_hierarchy_exports_core_entrypoints() -> None:
    import dresg.inference as inference
    import dresg.models.diffusion as diffusion
    import dresg.models.gs as gs
    import dresg.training as training
    from dresg.data.cameras import build_scaled_cameras
    from dresg.inference.paths import VideoPathRequest, build_video_path
    from dresg.models.gs.fitting import (
        DinoPatchContentLoss,
        compute_appearance_losses,
    )
    from dresg.models.gs.rendering.rasterization import rasterize_gaussians
    from dresg.utils.view_selection.workflow import run_view_selection

    assert inference.__all__ == (
        "export_train_view_renders",
        "render_run_train_views",
        "render_run_video",
        "render_scene_video",
    )
    assert callable(inference.export_train_view_renders)
    assert callable(inference.render_run_train_views)
    assert callable(inference.render_run_video)
    assert callable(inference.render_scene_video)
    assert diffusion.__all__ == (
        "DiffusionGuidance",
        "build_diffusion_guidance",
    )
    assert callable(diffusion.DiffusionGuidance)
    assert callable(diffusion.build_diffusion_guidance)
    assert not hasattr(diffusion, "DiffusionTeacher")
    assert gs.__all__ == ("GaussianScene", "build_gaussian_scene")
    assert callable(gs.GaussianScene)
    assert callable(gs.build_gaussian_scene)
    assert training.__all__ == ("DReSGTrainer",)
    assert callable(training.DReSGTrainer)
    assert callable(build_scaled_cameras)
    assert callable(rasterize_gaussians)
    assert callable(compute_appearance_losses)
    assert callable(DinoPatchContentLoss)
    assert callable(VideoPathRequest)
    assert callable(build_video_path)
    assert callable(run_view_selection)


def test_old_pipelines_module_is_removed() -> None:
    def _find_spec(name: str):
        try:
            return importlib.util.find_spec(name)
        except ModuleNotFoundError:
            return None

    assert _find_spec("dresg.pipelines." + "a" + "d") is None
    assert _find_spec("dresg.pipelines.config") is None
    assert _find_spec("dresg.pipelines.consensus") is None
    assert _find_spec("dresg.pipelines.stage") is None
    assert _find_spec("dresg.pipelines.views") is None
    assert _find_spec("dresg.pipelines." + "rendering") is None


def test_replaced_flat_modules_are_removed() -> None:
    def _find_spec(name: str):
        try:
            return importlib.util.find_spec(name)
        except ModuleNotFoundError:
            return None

    removed = (
        "dresg.data.traj",
        "dresg.models.diffusion.attention.guidance",
        "dresg.models.diffusion.codec",
        "dresg.models.diffusion.inputs",
        "dresg.models.diffusion.projection_feedback",
        "dresg.models.diffusion.residual_schedule",
        "dresg.models.diffusion.runtime",
        "dresg.models.diffusion.scheduling.teacher_scale",
        "dresg.models.diffusion.teacher_images",
        "dresg.models.diffusion.latents.view_bank",
        "dresg.models.gs.appearance",
        "dresg.models.gs.appearance_gradients",
        "dresg.models.gs.checkpoint",
        "dresg.models.gs.content_losses",
        "dresg.models.gs.direct_rgb_scene",
        "dresg.models.gs.gradient_fusion",
        "dresg.models.gs.image_losses",
        "dresg.models.gs.losses",
        "dresg.models.gs.ply_conversion",
        "dresg.models.gs.rasterization",
        "dresg.models.gs.render",
        "dresg.models.gs.sh",
        "dresg.inference.camera_config",
        "dresg.inference.camera_paths",
        "dresg.inference.llff_paths",
        "dresg.inference.path_math",
        "dresg.inference.paths.artifact",
        "dresg.inference.paths.build",
        "dresg.inference.tnt_paths",
        "dresg.inference.video_render",
        "dresg.rendering",
        "dresg.training.appearance",
        "dresg.training.appearance_fit",
        "dresg.training.artifacts",
        "dresg.training.color",
        "dresg.training.color_transfer",
        "dresg.training.fit",
        "dresg.training.metrics",
        "dresg.training.preflight",
        "dresg.training.guidance_references",
        "dresg.training.guidance_setup",
        "dresg.training.progress",
        "dresg.training.records",
        "dresg.training.scene_setup",
        "dresg.training.setup",
        "dresg.training.stage",
        "dresg.training.stages.gs",
        "dresg.training.teacher_fit",
        "dresg.training.artifacts.checkpoint",
        "dresg.training.artifacts.finalize",
        "dresg.training.artifacts.render_export",
        "dresg.training.checkpointing",
        "dresg.training.render_export",
        "dresg.training.view_update",
        "dresg.models.diffusion.teacher",
        "dresg.results",
        "dresg.view_selection",
    )
    assert not [name for name in removed if _find_spec(name) is not None]
