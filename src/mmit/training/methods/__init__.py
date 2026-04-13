"""Training methods for multimodal fine-tuning.

Classic methods:
  - QLoRA, LoRA          — parameter-efficient LoRA variants
  - DoRA                 — weight-decomposed LoRA (Liu et al. ICML 2024)
  - RandLoRA             — full-rank LoRA via random bases (ICLR 2025)
  - FullFT               — full fine-tuning (all parameters)
  - FreezeTuning         — train selected modules only

Paper methods:
  - L2T          — instruction-aware loss masking (Zhou et al. 2025)
  - MoReS        — representation steering (Bi et al. 2025)
  - LoRAInLoRA   — nested LoRA for continual learning (Che et al. 2025)
  - LLaVA-MoLE  — sparse mixture of LoRA experts (Chen et al. 2024)
  - ReFT/LoReFT — representation finetuning (Wu et al. NeurIPS 2024)
"""

from mmit.training.methods.lora import QLoRAMethod, LoRAMethod
from mmit.training.methods.dora import DoRAMethod
from mmit.training.methods.randlora import RandLoRAMethod
from mmit.training.methods.full_ft import FullFTMethod
from mmit.training.methods.freeze import FreezeTuningMethod
from mmit.training.methods.l2t import L2TMethod
from mmit.training.methods.mores import MoReSMethod
from mmit.training.methods.lora_in_lora import LoRAInLoRAMethod
from mmit.training.methods.mole import MoLEMethod
from mmit.training.methods.reft import ReFTMethod

__all__ = [
    "QLoRAMethod",
    "LoRAMethod",
    "DoRAMethod",
    "RandLoRAMethod",
    "FullFTMethod",
    "FreezeTuningMethod",
    "L2TMethod",
    "MoReSMethod",
    "LoRAInLoRAMethod",
    "MoLEMethod",
    "ReFTMethod",
]
