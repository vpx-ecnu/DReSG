from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from dresg.config import DReSGConfig, to_typed_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "conf"


def make_config(overrides: dict[str, object] | None = None) -> DReSGConfig:
    """Materialize repository YAML defaults for isolated unit tests."""
    values = OmegaConf.load(CONFIG_ROOT / "config.yaml")
    for key in ("defaults", "hydra"):
        del values[key]
    values.data.base_ply = "/tmp/base.ply"
    values.data.scene_dir = "/tmp/scene"
    values.data.style_image = "/tmp/style.png"
    values.data.output_dir = "/tmp/output"
    values.data.views = [0]
    config = OmegaConf.merge(OmegaConf.structured(DReSGConfig), values)
    if overrides:
        config = OmegaConf.merge(config, overrides)
    return to_typed_config(config)
