"""Public Gaussian-scene API with a lazy implementation import."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ("GaussianScene", "build_gaussian_scene")

if TYPE_CHECKING:
    from dresg.models.gs.scene import GaussianScene, build_gaussian_scene


def __getattr__(name: str) -> Any:
    if name in __all__:
        from dresg.models.gs.scene import GaussianScene, build_gaussian_scene

        return {
            "GaussianScene": GaussianScene,
            "build_gaussian_scene": build_gaussian_scene,
        }[name]
    raise AttributeError(name)
