"""CEPlusOrthoLoss — cross-entropy + orthogonal penalty.

Extracted from MoReS's compute_loss. The orthogonal penalty encourages
the intervention matrices to remain orthogonal (||R R^T - I||_F^2).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from mmit.training.losses.base import LossFunction
from mmit.training.losses.ce_loss import CrossEntropyLoss


class CEPlusOrthoLoss(LossFunction):
    """CE loss + orthogonal regularization penalty for MoReS interventions."""

    def __init__(self, ortho_weight: float = 0.01, **kwargs):
        self.ortho_weight = ortho_weight
        self._ce = CrossEntropyLoss()

    def compute(
        self,
        model: Any,
        batch: Dict[str, Any],
        outputs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        text_loss, _ = self._ce.compute(model, batch, outputs)
        metrics = {"text_loss": text_loss.item()}

        # Collect orthogonal penalties from all MoReS interventions
        ortho_loss = torch.tensor(0.0, device=text_loss.device, dtype=text_loss.dtype)
        for module in model.modules():
            if hasattr(module, "orthogonal_penalty"):
                ortho_loss = ortho_loss + module.orthogonal_penalty()

        if ortho_loss.item() > 0:
            total_loss = text_loss + self.ortho_weight * ortho_loss
            metrics["ortho_loss"] = ortho_loss.item()
        else:
            total_loss = text_loss

        return total_loss, metrics
