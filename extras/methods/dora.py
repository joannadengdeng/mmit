"""DoRA: Weight-Decomposed Low-Rank Adaptation.

Moved to extras/ — not included in the published mmit package.
To use: copy this file to src/mmit/training/methods/ and register in __init__.py.

Reference: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024.
"""
from mmit.training.methods.lora import LoRAMethod


class DoRAMethod(LoRAMethod):
    """DoRA: Weight-Decomposed Low-Rank Adaptation."""

    name = "dora"
    display_name = "DoRA"
    paper_ref = "Liu et al., ICML 2024"

    def _lora_kwargs(self):
        return {"use_dora": True}
