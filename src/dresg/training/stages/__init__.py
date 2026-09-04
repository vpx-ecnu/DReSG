"""Training-stage execution boundary."""

from dresg.training.stages.color import ColorStage
from dresg.training.stages.feedback import FeedbackStage
from dresg.training.stages.guidance import GuidanceStage

__all__ = (
    "ColorStage",
    "FeedbackStage",
    "GuidanceStage",
)
