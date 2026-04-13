"""DoRA: Weight-Decomposed Low-Rank Adaptation.

Paper: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024
arXiv: 2402.09353

Key idea: Decompose pretrained weights into magnitude and direction components.
LoRA is applied only to the directional matrix, while the magnitude vector is
trained separately.  This mimics full fine-tuning's learning pattern more closely
than standard LoRA, yielding ~0.7% average improvement with zero extra inference
overhead (magnitude + direction can be merged back).
"""
from mmit.training.methods.lora import LoRAMethod


class DoRAMethod(LoRAMethod):
    """DoRA: Weight-Decomposed Low-Rank Adaptation."""

    name = "dora"
    display_name = "DoRA"
    paper_ref = "Liu et al., ICML 2024"

    def _lora_kwargs(self):
        return {"use_dora": True}
