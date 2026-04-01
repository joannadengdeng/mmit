"""LoRA / QLoRA / DoRA fine-tuning methods.

These three methods share the same core logic (PEFT LoraConfig), differing only
in quantization (QLoRA) and weight decomposition (DoRA).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmit.training.methods.base import TrainingMethod
from mmit.training.losses.ce_loss import CrossEntropyLoss

IGNORE_INDEX = -100

_ce_loss = CrossEntropyLoss()


def _auto_detect_targets(
    model: nn.Module,
    freeze_patterns: List[str] = (),
) -> list[str]:
    """Find common LoRA target module names, excluding frozen modules.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.
    freeze_patterns : list of str
        Module name patterns to exclude from LoRA injection.
        E.g. ["vision_tower", "vision_model"] prevents LoRA on vision encoder.
    """
    targets = set()
    for name, _ in model.named_modules():
        # Skip modules matching freeze patterns
        if freeze_patterns and any(p in name for p in freeze_patterns):
            continue
        short = name.split(".")[-1]
        if short in ("q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"):
            targets.add(short)
    return sorted(targets) if targets else ["q_proj", "v_proj"]


# Keep for backward compat — other methods import this
def _standard_ce_loss(outputs, batch):
    """Standard causal LM cross-entropy loss."""
    loss, _ = _ce_loss.compute(None, batch, outputs)
    return loss


class LoRAMethod(TrainingMethod):
    """Standard LoRA fine-tuning (bf16 precision)."""

    name = "lora"
    display_name = "LoRA"
    paper_ref = "Hu et al., ICLR 2022"

    def default_config(self):
        return {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],       # empty = auto-detect
            "modules_to_save": [],      # e.g. ["multi_modal_projector"]
            "freeze_patterns": [],      # passed to base class
        }

    def ui_params(self):
        return [
            {"name": "lora_r", "type": "slider", "label": "LoRA Rank (r)",
             "default": 8, "min": 4, "max": 64, "step": 4},
            {"name": "lora_alpha", "type": "slider", "label": "LoRA Alpha",
             "default": 16, "min": 8, "max": 128, "step": 8},
            {"name": "lora_dropout", "type": "number", "label": "Dropout",
             "default": 0.05},
        ]

    def _lora_kwargs(self) -> dict:
        """Extra kwargs for LoraConfig. Override in subclasses."""
        return {}

    def _prepare_model_impl(self, model, processor, config):
        from peft import LoraConfig, get_peft_model, TaskType

        r = int(config.get("lora_r", 8))
        alpha = int(config.get("lora_alpha", 16))
        dropout = float(config.get("lora_dropout", 0.05))
        freeze_patterns = config.get("freeze_patterns", [])

        # Target modules: user-specified or auto-detected (respecting freeze_patterns)
        targets = config.get("target_modules") or _auto_detect_targets(model, freeze_patterns)

        # modules_to_save: full-parameter training for specified modules (e.g. projector)
        modules_to_save = config.get("modules_to_save") or None

        lora_config = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets,
            modules_to_save=modules_to_save,
            task_type=TaskType.CAUSAL_LM,
            **self._lora_kwargs(),
        )
        peft_model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        info = (
            f"{self.display_name}: r={r}, alpha={alpha}, dropout={dropout}\n"
            f"Target modules: {targets}\n"
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
        if modules_to_save:
            info += f"\nModules to save (full train): {modules_to_save}"
        return peft_model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        metadata["ft_method"] = self.name
        with open(os.path.join(path, "mmit_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoVLM.from_pretrained(
            base_model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, path)
        model.eval()
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} ({self.display_name}: {adapter_name})"}
        return model, processor, info


class QLoRAMethod(LoRAMethod):
    """QLoRA: LoRA with 4-bit quantized base model."""

    name = "qlora"
    display_name = "QLoRA"
    paper_ref = "Dettmers et al., NeurIPS 2023"

    def requires_quantization(self):
        return True


