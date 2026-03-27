"""Training methods for multimodal fine-tuning.

Classic methods:
  - QLoRA, LoRA, DoRA  — parameter-efficient LoRA variants
  - FullFT              — full fine-tuning (all parameters)
  - FreezeTuning        — train selected modules only

Paper methods:
  - L2T          — instruction-aware loss masking (Zhou et al. 2025)
  - MoReS        — representation steering (Bi et al. 2025)
  - LoRAInLoRA   — nested LoRA for continual learning (Che et al. 2025)
"""

from mmit.training.methods.lora import QLoRAMethod, LoRAMethod, DoRAMethod
from mmit.training.methods.full_ft import FullFTMethod
from mmit.training.methods.freeze import FreezeTuningMethod
from mmit.training.methods.l2t import L2TMethod
from mmit.training.methods.mores import MoReSMethod
from mmit.training.methods.lora_in_lora import LoRAInLoRAMethod

__all__ = [
    "QLoRAMethod",
    "LoRAMethod",
    "DoRAMethod",
    "FullFTMethod",
    "FreezeTuningMethod",
    "L2TMethod",
    "MoReSMethod",
    "LoRAInLoRAMethod",
]
