"""RandLoRA: Full-rank parameter-efficient fine-tuning.

Paper: Kalajdzievski, "RandLoRA: Full-rank parameter-efficient fine-tuning
       of large models", ICLR 2025
arXiv: 2502.00987

Key idea: Perform full-rank weight updates using learned linear combinations of
fixed, low-rank random matrices.  Only diagonal scaling matrices are trainable.
This overcomes LoRA's fundamental rank deficiency while keeping the same parameter
and memory efficiency during training.  Particularly beneficial for vision-language
tasks where full-rank updates significantly close the gap to full fine-tuning.
"""
from __future__ import annotations

from typing import Any, Dict, List

from mmit.training.methods.lora import LoRAMethod


class RandLoRAMethod(LoRAMethod):
    """RandLoRA: full-rank adaptation via random basis matrices."""

    name = "randlora"
    display_name = "RandLoRA"
    paper_ref = "Kalajdzievski, ICLR 2025"

    def default_config(self):
        cfg = super().default_config()
        cfg["lora_r"] = 8
        return cfg

    def _lora_kwargs(self) -> dict:
        return {"use_rslora": True}
