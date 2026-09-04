"""Gaussian fitting losses, resources, and gradient fusion."""

from dresg.models.gs.fitting.appearance import (
    AppearanceLosses,
    compute_appearance_losses,
)
from dresg.models.gs.fitting.dino import DinoPatchContentLoss
from dresg.models.gs.fitting.fusion import (
    APPEARANCE_UPDATE_RULES,
    fuse_appearance_gradients,
)

__all__ = [
    "APPEARANCE_UPDATE_RULES",
    "AppearanceLosses",
    "DinoPatchContentLoss",
    "compute_appearance_losses",
    "fuse_appearance_gradients",
]
