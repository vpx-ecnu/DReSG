"""Lazy public boundary for the DReSG training lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ("DReSGTrainer",)

if TYPE_CHECKING:
    from dresg.training.trainer import DReSGTrainer


def __getattr__(name: str) -> Any:
    if name == "DReSGTrainer":
        from dresg.training.trainer import DReSGTrainer

        return DReSGTrainer
    raise AttributeError(name)
