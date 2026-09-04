from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import MISSING, DictConfig, MissingMandatoryValue, OmegaConf

from dresg.config import (
    DataConfig,
    DReSGConfig,
    register_dresg_config,
    to_typed_config,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "conf"

register_dresg_config()


def test_repository_hydra_configs_exist() -> None:
    assert (CONFIG_ROOT / "config.yaml").is_file()
    assert (CONFIG_ROOT / "view_selection" / "llff" / "fern.yaml").is_file()
    assert (CONFIG_ROOT / "view_selection" / "tnt" / "m60.yaml").is_file()


def test_base_yaml_is_the_single_source_of_values() -> None:
    payload = yaml.safe_load((CONFIG_ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert payload["defaults"][:2] == ["dresg_schema", "_self_"]
    assert payload["defaults"][-1] == {"override hydra/job_logging": "disabled"}
    assert payload["data"]["base_ply"] == "???"
    assert DataConfig.__dataclass_fields__["factor"].default == MISSING
    assert "method" not in payload["view_selection"]
    assert "snr_balance_multiplier" not in payload["teacher"]
    assert "geometry_constraint_mode" not in payload["appearance_optim"]
    assert "scale_space" not in payload["teacher"]
    assert "content_source" not in payload["teacher"]
    assert "content3d_mode" not in payload["image_loss"]
    assert "final_prefix_fit_steps" not in payload["schedule"]
    assert "log_every" not in payload["schedule"]
    assert "save_debug" not in payload["color_transfer"]
    assert "stage_image_mode" not in payload["artifacts"]
    assert "save_setup_images" not in payload["artifacts"]
    assert "model" not in payload
    assert payload["artifacts"] == {
        "save_train_views": True,
        "save_style_image": True,
        "video": {
            "path": None,
            "fps": 30,
            "batch_size": 1,
        },
    }
    assert payload["hydra"]["output_subdir"] == ".hydra"
    assert payload["hydra"]["run"]["dir"] == "${data.output_dir}"
    assert payload["hydra"]["sweep"]["dir"] == "${data.output_dir}"


def test_hydra_composes_and_materializes_typed_smoke_config() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        raw_config = compose(
            config_name="config",
            overrides=[
                "experiment=smoke_test",
                "schedule.prefixes=[1,2]",
                "data.views=[3,5]",
            ],
        )

    assert isinstance(raw_config, DictConfig)
    assert OmegaConf.get_type(raw_config) is DReSGConfig
    config = to_typed_config(raw_config)
    assert isinstance(config, DReSGConfig)
    assert config.schedule.prefixes == [1, 2]
    assert config.data.views == [3, 5]
    assert isinstance(config.data.base_ply, Path)
    assert not hasattr(config.data, "base_ckpt")
    assert not hasattr(config.data, "image_pyramid_layout")
    assert not hasattr(config.artifacts, "save_checkpoint")
    assert not hasattr(config.artifacts, "save_video")
    assert not hasattr(config.view_selection, "method")
    assert not hasattr(config.teacher, "snr_balance_multiplier")
    assert not hasattr(config.appearance_optim, "geometry_constraint_mode")
    assert not hasattr(config.teacher, "scale_space")
    assert not hasattr(config.teacher, "content_source")
    assert not hasattr(config.image_loss, "content3d_mode")
    assert not hasattr(config.schedule, "final_prefix_fit_steps")
    assert not hasattr(config.schedule, "log_every")
    assert not hasattr(config.color_transfer, "save_debug")
    assert not hasattr(config.artifacts, "stage_image_mode")
    assert not hasattr(config.artifacts, "save_setup_images")


def test_required_yaml_value_fails_during_materialization() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        config = compose(config_name="config")

    with pytest.raises(MissingMandatoryValue):
        to_typed_config(config)


def test_schema_rejects_unknown_override_key() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "schedule.fit_stepz=3"],
        )


@pytest.mark.parametrize(
    "removed_override",
    [
        "view_selection.method=manual",
        "teacher.snr_balance_multiplier=4.0",
        "appearance_optim.geometry_constraint_mode=projected_box",
        "artifacts.save_final_ckpt=false",
        "artifacts.render_video=false",
        "artifacts.export_train_views=false",
        "data.base_ckpt=/tmp/base.pt",
        "artifacts.save_checkpoint=false",
        "artifacts.save_video=false",
        "data.image_pyramid_layout=llff_indexed",
    ],
)
def test_schema_rejects_removed_configuration_keys(removed_override: str) -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", removed_override],
        )


def test_schema_rejects_removed_teacher_content_source() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "teacher.content_source=render"],
        )


def test_schema_rejects_removed_content3d_mode() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "image_loss.content3d_mode=none"],
        )


def test_schema_rejects_removed_final_prefix_fit_steps() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "schedule.final_prefix_fit_steps=10"],
        )


def test_schema_rejects_removed_log_interval() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "schedule.log_every=10"],
        )


def test_schema_rejects_removed_stage_metrics_mode() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "debug.stage_metrics_mode=full"],
        )


def test_schema_rejects_wrong_override_type() -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)),
        pytest.raises(ConfigCompositionException),
    ):
        compose(
            config_name="config",
            overrides=["experiment=smoke_test", "schedule.fit_steps=wrong"],
        )


def test_hydra_view_preset_overrides_required_views() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        raw_config = compose(
            config_name="config",
            overrides=[
                "view_selection=llff/fern",
                "data.base_ply=/tmp/base.ply",
                "data.scene_dir=/tmp/scene",
                "data.style_image=/tmp/style.png",
                "data.output_dir=/tmp/output",
            ],
        )

    config = to_typed_config(raw_config)
    assert config.view_selection.scene == "fern"
    assert len(config.data.views) == config.view_selection.selected_count


def test_tnt_experiment_only_overrides_dataset_specific_values() -> None:
    payload = yaml.safe_load(
        (CONFIG_ROOT / "experiment" / "tnt.yaml").read_text(encoding="utf-8")
    )

    assert set(payload) == {"data", "guidance", "artifacts"}
    assert payload["data"] == {"factor": 1}
    assert payload["guidance"] == {"attention": {"content_weight": 0.1}}
    assert payload["artifacts"]["video"] == {"batch_size": 8}
