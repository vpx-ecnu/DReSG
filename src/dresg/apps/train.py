"""Unified Hydra entrypoint for the DReSG stylization pipeline."""

from __future__ import annotations

from pathlib import Path

try:
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Hydra entrypoint requires hydra-core. Install it with `pip install hydra-core` "
        "or install this repo's updated requirements.txt."
    ) from exc

from dresg.config import register_dresg_config, to_typed_config

register_dresg_config()


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(raw_config: DictConfig) -> None:
    """Validate, materialize, and run one composed Hydra configuration."""
    from dresg.training import DReSGTrainer

    config = to_typed_config(raw_config)
    config.data.output_dir = Path(HydraConfig.get().runtime.output_dir)
    DReSGTrainer(config).run()


if __name__ == "__main__":
    main()
