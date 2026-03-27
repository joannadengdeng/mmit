"""LossFunction ABC — decoupled loss computation.

Separates "what loss to compute" from "how to prepare the model" (TrainingMethod).
This allows combining any PEFT strategy with any loss function.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class LossFunction(ABC):
    """Base class for all loss functions."""

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        outputs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the training loss.

        Parameters
        ----------
        model : nn.Module
            The prepared model.
        batch : dict
            Training batch with input_ids, labels, attention_mask, etc.
        outputs :
            Model forward output (has .loss, .logits, etc.).

        Returns
        -------
        (loss, metrics_dict) — scalar loss and optional logging metrics.
        """

    def on_prepare(self, model: nn.Module, config: Dict[str, Any]) -> None:
        """Called after model preparation. Use to attach auxiliary heads, hooks, etc.

        For example, CEPlusReconLoss attaches a denoising head here.
        Default: no-op.
        """
        pass
