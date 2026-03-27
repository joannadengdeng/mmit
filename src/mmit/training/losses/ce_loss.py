"""CrossEntropyLoss — standard causal LM cross-entropy.

Extracted from the original _standard_ce_loss in lora.py.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from mmit.training.losses.base import LossFunction

IGNORE_INDEX = -100


class CrossEntropyLoss(LossFunction):
    """Standard causal language modeling cross-entropy loss."""

    def compute(
        self,
        model: Any,
        batch: Dict[str, Any],
        outputs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss, {}

        logits = outputs.logits
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return loss, {}
