from __future__ import annotations

import pytest

from dresg.training.trainer import DReSGTrainer
from dresg.training.validation import validate_run_config
from tests.config_factory import make_config


def valid_config():
    return make_config()


def set_config_value(config, dotted_name: str, value) -> None:
    owner = config
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)


class TestRunPreflight:
    def test_path_fields_reject_string_coercion(self) -> None:
        cfg = valid_config()
        cfg.data.base_ply = ""

        with pytest.raises(TypeError, match="data.base_ply must be a pathlib.Path"):
            validate_run_config(cfg, check_paths=False)

    def test_base_ply_requires_canonical_suffix(self) -> None:
        cfg = valid_config()
        cfg.data.base_ply = cfg.data.base_ply.with_suffix(".pt")

        with pytest.raises(ValueError, match="data.base_ply must use the .ply suffix"):
            validate_run_config(cfg, check_paths=False)

    def test_missing_inputs_fail_before_output_directory_is_created(
        self,
        tmp_path,
    ) -> None:
        cfg = valid_config()
        cfg.data.base_ply = str(tmp_path / "missing.ply")
        cfg.data.scene_dir = str(tmp_path / "missing_scene")
        cfg.data.style_image = str(tmp_path / "missing_style.png")
        output_dir = tmp_path / "output"
        cfg.data.output_dir = str(output_dir)

        with pytest.raises(TypeError, match="data.base_ply"):
            DReSGTrainer(cfg)

        assert not output_dir.exists()

    def test_nonempty_output_fails_at_preflight(self, tmp_path) -> None:
        cfg = valid_config()
        cfg.data.base_ply = tmp_path / "base.ply"
        cfg.data.base_ply.touch()
        cfg.data.scene_dir = tmp_path / "scene"
        cfg.data.scene_dir.mkdir()
        cfg.data.style_image = tmp_path / "style.png"
        cfg.data.style_image.touch()
        cfg.data.output_dir = tmp_path / "output"
        cfg.data.output_dir.mkdir()
        (cfg.data.output_dir / "stale-output.txt").touch()

        with pytest.raises(FileExistsError, match="contains stale artifacts"):
            validate_run_config(cfg)

    def test_current_hydra_metadata_is_allowed_at_preflight(self, tmp_path) -> None:
        cfg = valid_config()
        cfg.data.base_ply = tmp_path / "base.ply"
        cfg.data.base_ply.touch()
        cfg.data.scene_dir = tmp_path / "scene"
        cfg.data.scene_dir.mkdir()
        cfg.data.style_image = tmp_path / "style.png"
        cfg.data.style_image.touch()
        cfg.data.output_dir = tmp_path / "output"
        hydra_dir = cfg.data.output_dir / ".hydra"
        hydra_dir.mkdir(parents=True)
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            (hydra_dir / name).touch()

        validate_run_config(cfg)

    def test_incomplete_hydra_metadata_is_rejected(self, tmp_path) -> None:
        cfg = valid_config()
        cfg.data.base_ply = tmp_path / "base.ply"
        cfg.data.base_ply.touch()
        cfg.data.scene_dir = tmp_path / "scene"
        cfg.data.scene_dir.mkdir()
        cfg.data.style_image = tmp_path / "style.png"
        cfg.data.style_image.touch()
        cfg.data.output_dir = tmp_path / "output"
        hydra_dir = cfg.data.output_dir / ".hydra"
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "config.yaml").touch()

        with pytest.raises(FileExistsError, match="invalid Hydra metadata"):
            validate_run_config(cfg)

    @pytest.mark.parametrize(
        ("invalid", "error"),
        [
            (True, TypeError),
            (42.0, TypeError),
            (-1, ValueError),
            (2**32, ValueError),
        ],
    )
    def test_runtime_seed_is_strict(
        self,
        invalid,
        error: type[Exception],
    ) -> None:
        cfg = valid_config()
        cfg.runtime.seed = invalid

        with pytest.raises(error, match="runtime.seed"):
            validate_run_config(cfg, check_paths=False)

    @pytest.mark.parametrize(
        ("field", "invalid"),
        [
            ("teacher_width", True),
            ("teacher_height", 320.0),
        ],
    )
    def test_teacher_dimensions_require_strict_integers(
        self,
        field: str,
        invalid,
    ) -> None:
        cfg = valid_config()
        setattr(cfg.rendering, field, invalid)

        with pytest.raises(TypeError, match=f"rendering.{field}"):
            validate_run_config(cfg, check_paths=False)

    def test_teacher_dimensions_require_multiples_of_64(self) -> None:
        cfg = valid_config()
        cfg.rendering.teacher_width = 450

        with pytest.raises(ValueError, match="multiples of 64"):
            validate_run_config(cfg, check_paths=False)

    def test_geometry_delta_limits_must_be_positive(self) -> None:
        cfg = valid_config()
        cfg.appearance_optim.max_mean_delta = 0.0

        with pytest.raises(ValueError, match="appearance_optim.max_mean_delta"):
            validate_run_config(cfg, check_paths=False)

    def test_quaternion_optimization_requires_geometry(self) -> None:
        cfg = valid_config()
        cfg.appearance_optim.optimize_geometry = False
        cfg.appearance_optim.optimize_geometry_quats = True

        with pytest.raises(ValueError, match="requires optimize_geometry"):
            validate_run_config(cfg, check_paths=False)

    def test_absent_video_path_disables_video(self) -> None:
        cfg = valid_config()
        cfg.artifacts.video.path = None

        validate_run_config(cfg, check_paths=False)

    def test_path_value_enables_video(self, tmp_path) -> None:
        cfg = valid_config()
        cfg.artifacts.video.path = tmp_path / "video_path.npz"

        validate_run_config(cfg, check_paths=False)

    def test_enabled_video_path_must_exist(self, tmp_path) -> None:
        cfg = valid_config()
        cfg.data.base_ply = tmp_path / "base.ply"
        cfg.data.base_ply.touch()
        cfg.data.scene_dir = tmp_path / "scene"
        cfg.data.scene_dir.mkdir()
        cfg.data.style_image = tmp_path / "style.png"
        cfg.data.style_image.touch()
        cfg.data.output_dir = tmp_path / "output"
        cfg.artifacts.video.path = tmp_path / "missing_video_path.npz"

        with pytest.raises(ValueError, match="artifacts.video.path does not exist"):
            validate_run_config(cfg)

    @pytest.mark.parametrize(
        ("field", "invalid", "error"),
        [
            ("fps", True, TypeError),
            ("fps", 0, ValueError),
            ("batch_size", False, TypeError),
            ("batch_size", 0, ValueError),
        ],
    )
    def test_video_integer_fields_are_strict(
        self,
        field: str,
        invalid,
        error: type[Exception],
    ) -> None:
        cfg = valid_config()
        setattr(cfg.artifacts.video, field, invalid)

        with pytest.raises(error, match=f"artifacts.video.{field}"):
            validate_run_config(cfg, check_paths=False)

    def test_video_path_type_is_strict(self) -> None:
        cfg = valid_config()
        cfg.artifacts.video.path = "/tmp/path.npz"

        with pytest.raises(TypeError, match="artifacts.video.path"):
            validate_run_config(cfg, check_paths=False)

    def test_post_color_transfer_is_supported(self) -> None:
        cfg = valid_config()
        cfg.color_transfer.post_enabled = True

        validate_run_config(cfg, check_paths=False)

    def test_feedback_none_is_valid(self) -> None:
        cfg = valid_config()
        cfg.guidance.feedback.mode = "none"

        validate_run_config(cfg, check_paths=False)

    def test_unknown_feedback_mode_is_rejected(self) -> None:
        cfg = valid_config()
        cfg.guidance.feedback.mode = "blend"

        with pytest.raises(ValueError, match="guidance.feedback.mode"):
            validate_run_config(cfg, check_paths=False)

    def test_empty_diffusion_model_id_is_rejected(self) -> None:
        cfg = valid_config()
        cfg.guidance.backbone.model_id = " "

        with pytest.raises(ValueError, match="guidance.backbone.model_id"):
            validate_run_config(cfg, check_paths=False)

    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_guidance_values_are_rejected(self, invalid: float) -> None:
        cfg = valid_config()
        cfg.guidance.optimization.learning_rate = invalid

        with pytest.raises(ValueError, match="must be finite"):
            validate_run_config(cfg, check_paths=False)

    @pytest.mark.parametrize(
        "field",
        [
            "data.factor",
            "schedule.fit_steps",
            "color_transfer.post_fit_steps",
            "image_loss.content3d_dino_size",
            "guidance.optimization.num_inference_steps",
            "guidance.optimization.inner_iterations",
            "guidance.optimization.view_batch_size",
        ],
    )
    @pytest.mark.parametrize("invalid", [True, 1.0])
    def test_integer_domains_reject_coercion(
        self,
        field: str,
        invalid,
    ) -> None:
        cfg = valid_config()
        set_config_value(cfg, field, invalid)

        with pytest.raises(TypeError, match=field):
            validate_run_config(cfg, check_paths=False)

    @pytest.mark.parametrize(
        "field",
        [
            "rendering.render_scale",
            "teacher.scale",
            "appearance_optim.lr",
            "image_loss.lambda_l1",
            "guidance.attention.content_weight",
            "guidance.optimization.learning_rate",
        ],
    )
    def test_numeric_domains_reject_boolean_values(self, field: str) -> None:
        cfg = valid_config()
        set_config_value(cfg, field, True)

        with pytest.raises(TypeError, match=field):
            validate_run_config(cfg, check_paths=False)

    @pytest.mark.parametrize("views", [[False], [0.0], ["0"]])
    def test_view_ids_reject_coercion(self, views: list[object]) -> None:
        cfg = valid_config()
        cfg.data.views = views

        with pytest.raises(TypeError, match="data.views entries"):
            validate_run_config(cfg, check_paths=False)

    def test_prefixes_reject_boolean_values(self) -> None:
        cfg = valid_config()
        cfg.schedule.prefixes = [True]

        with pytest.raises(TypeError, match="schedule.prefixes entries"):
            validate_run_config(cfg, check_paths=False)

    def test_max_stages_uses_none_or_a_bounded_positive_integer(self) -> None:
        cfg = valid_config()
        cfg.schedule.max_stages = None
        validate_run_config(cfg, check_paths=False)

        cfg.schedule.max_stages = 0
        with pytest.raises(ValueError, match="schedule.max_stages must be positive"):
            validate_run_config(cfg, check_paths=False)

        cfg.schedule.max_stages = len(cfg.schedule.prefixes) + 1
        with pytest.raises(ValueError, match="must not exceed"):
            validate_run_config(cfg, check_paths=False)
