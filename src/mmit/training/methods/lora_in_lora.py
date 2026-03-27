"""LoRA-in-LoRA: nested LoRA for continual visual instruction tuning.

Paper: Che et al., "LoRA in LoRA: Towards Parameter-Efficient Architecture
Expansion for Continual Visual Instruction Tuning", 2025

Key idea: For continual learning across sequential tasks, merge the previous
task's LoRA into base weights, then apply a new (smaller) LoRA on top.
This prevents catastrophic forgetting while keeping parameter efficiency.

Usage:
  Stage 1 (task 1): standard LoRA
  Stage 2 (task 2): lora_in_lora with outer_checkpoint = stage 1 checkpoint
  Stage 3 (task 3): lora_in_lora with outer_checkpoint = stage 2 checkpoint
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from mmit.training.methods.base import TrainingMethod
from mmit.training.methods.lora import _auto_detect_targets
from mmit.training.losses.ce_loss import CrossEntropyLoss

_ce_loss = CrossEntropyLoss()


class LoRAInLoRAMethod(TrainingMethod):
    """Nested LoRA: merge outer LoRA into base, apply inner LoRA on top."""

    name = "lora_in_lora"
    display_name = "LoRA-in-LoRA"
    paper_ref = "Che et al., 2025"

    def __init__(self):
        self._outer_checkpoint = None

    def default_config(self):
        return {
            "outer_checkpoint": "",     # path to previous task's LoRA checkpoint
            "inner_lora_r": 4,          # inner LoRA rank (typically smaller)
            "inner_lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": [],
            "freeze_patterns": [],
        }

    def ui_params(self):
        return [
            {"name": "outer_checkpoint", "type": "text",
             "label": "Outer LoRA checkpoint path", "default": ""},
            {"name": "inner_lora_r", "type": "slider", "label": "Inner LoRA Rank",
             "default": 4, "min": 1, "max": 32, "step": 1},
            {"name": "inner_lora_alpha", "type": "slider", "label": "Inner LoRA Alpha",
             "default": 8, "min": 4, "max": 64, "step": 4},
        ]

    def _prepare_model_impl(self, model, processor, config):
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType

        outer_checkpoint = config.get("outer_checkpoint", "")
        inner_r = int(config.get("inner_lora_r", 4))
        inner_alpha = int(config.get("inner_lora_alpha", 8))
        dropout = float(config.get("lora_dropout", 0.05))
        freeze_patterns = config.get("freeze_patterns", [])

        info_parts = []

        # Step 1: Load and merge outer LoRA (becomes new base weights)
        if outer_checkpoint and os.path.isdir(outer_checkpoint):
            self._outer_checkpoint = outer_checkpoint
            model = PeftModel.from_pretrained(model, outer_checkpoint)
            model = model.merge_and_unload()
            info_parts.append(f"Outer LoRA merged from: {outer_checkpoint}")
        else:
            info_parts.append("No outer LoRA (first task)")

        # Step 2: Apply inner LoRA on the merged model
        targets = config.get("target_modules") or _auto_detect_targets(model, freeze_patterns)
        inner_config = LoraConfig(
            r=inner_r,
            lora_alpha=inner_alpha,
            lora_dropout=dropout,
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, inner_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info_parts.append(
            f"Inner LoRA: r={inner_r}, alpha={inner_alpha}\n"
            f"Target modules: {targets}\n"
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
        return model, "\n".join(info_parts)

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        metadata["ft_method"] = self.name
        if self._outer_checkpoint:
            metadata["outer_checkpoint"] = self._outer_checkpoint
        with open(os.path.join(path, "mmit_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
        from peft import PeftModel

        # Read metadata to find the chain of outer checkpoints
        meta_path = os.path.join(path, "mmit_meta.json")
        outer_checkpoint = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            outer_checkpoint = meta.get("outer_checkpoint")

        # Load base model
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoVLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

        # Recursively merge outer checkpoints
        if outer_checkpoint and os.path.isdir(outer_checkpoint):
            model = PeftModel.from_pretrained(model, outer_checkpoint)
            model = model.merge_and_unload()

        # Load inner LoRA
        model = PeftModel.from_pretrained(model, path)
        model.eval()
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

        info = {"model_id": f"{base_model_id} (LoRA-in-LoRA: {os.path.basename(path)})"}
        return model, processor, info
