"""Lazy public boundary for the run-specific diffusion model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ("DiffusionGuidance", "build_diffusion_guidance")

if TYPE_CHECKING:
    from dresg.models.diffusion.guidance import (
        DiffusionGuidance,
        build_diffusion_guidance,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        from dresg.models.diffusion.guidance import (
            DiffusionGuidance,
            build_diffusion_guidance,
        )

        return {
            "DiffusionGuidance": DiffusionGuidance,
            "build_diffusion_guidance": build_diffusion_guidance,
        }[name]
    raise AttributeError(name)
