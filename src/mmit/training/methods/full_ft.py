"""Full fine-tuning — update all model parameters.

Use ``freeze_patterns`` in method_params to selectively freeze modules:

    training_method: full_ft
    method_params:
      freeze_patterns: ["vision_tower"]   # freeze vision encoder
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from mmit.training.methods.base import TrainingMethod
from mmit.training.losses.ce_loss import CrossEntropyLoss

_ce_loss = CrossEntropyLoss()


class FullFTMethod(TrainingMethod):
    """Full fine-tuning: all parameters are trainable.

    Requires significantly more GPU memory than LoRA variants.
    Recommended only with large GPUs (>=24GB) or distributed training.
    Use ``freeze_patterns`` to selectively freeze modules (e.g. vision encoder).
    """

    name = "full_ft"
    display_name = "Full Fine-tuning"
    paper_ref = ""

    def default_config(self):
        return {"freeze_patterns": []}

    def ui_params(self):
        return []

    def _prepare_model_impl(self, model, processor, config):
        for p in model.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters())
        info = f"Full fine-tuning: all {trainable:,} parameters trainable"
        return model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        metadata["ft_method"] = self.name
        with open(os.path.join(path, "mmit_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoVLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model.eval()

        info = {"model_id": f"Full FT model from {path}"}
        return model, processor, info
