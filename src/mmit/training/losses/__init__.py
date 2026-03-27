"""Loss functions: decoupled loss computation for training."""

from mmit.training.losses.base import LossFunction
from mmit.training.losses.ce_loss import CrossEntropyLoss
from mmit.training.losses.ce_ortho import CEPlusOrthoLoss

__all__ = ["LossFunction", "CrossEntropyLoss", "CEPlusOrthoLoss"]
