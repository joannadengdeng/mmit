"""Freeze Tuning — freeze most of the model, train selected modules only.

Useful for VLMs where you want to update specific components:
  - Last N layers of the LLM
  - The vision-language projector
  - The LM head
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


def _find_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Find the sequential list of transformer layers in a model."""
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    return []


class FreezeTuningMethod(TrainingMethod):
    """Freeze Tuning: train only selected modules while freezing the rest.

    Supported train_modules:
      - "llm_last_n": last N transformer layers of the LLM
      - "projector": the vision-language projector (if accessible)
      - "lm_head": the language model output head
    """

    name = "freeze"
    display_name = "Freeze Tuning"
    paper_ref = ""

    def default_config(self):
        return {
            "train_modules": ["llm_last_n"],
            "num_layers": 4,
        }

    def ui_params(self):
        return [
            {"name": "train_modules", "type": "checkboxgroup",
             "label": "Modules to train",
             "choices": ["LLM last N layers", "Projector", "LM Head"],
             "default": ["LLM last N layers"]},
            {"name": "num_layers", "type": "slider",
             "label": "Last N layers to train",
             "default": 4, "min": 1, "max": 16, "step": 1},
        ]

    def _prepare_model_impl(self, model, processor, config):
        # Freeze everything first (skip quantized params that can't take grad)
        for p in model.parameters():
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = False

        train_modules = config.get("train_modules", ["LLM last N layers"])
        num_layers = int(config.get("num_layers", 4))
        unfrozen_parts = []

        def _unfreeze(module):
            for p in module.parameters():
                if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    p.requires_grad = True

        if "LLM last N layers" in train_modules:
            layers = _find_transformer_layers(model)
            if layers:
                for layer in layers[-num_layers:]:
                    _unfreeze(layer)
                unfrozen_parts.append(f"last {num_layers} LLM layers")

        if "LM Head" in train_modules:
            for name, module in model.named_modules():
                if "lm_head" in name:
                    _unfreeze(module)
                    unfrozen_parts.append("lm_head")
                    break

        if "Projector" in train_modules:
            for name, module in model.named_modules():
                if "multi_modal_projector" in name or "mm_projector" in name:
                    _unfreeze(module)
                    unfrozen_parts.append("projector")
                    break

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = (
            f"Freeze Tuning: training [{', '.join(unfrozen_parts)}]\n"
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        # Save only trainable parameters
        trained_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {k: v for k, v in model.state_dict().items() if k in trained_names}
        torch.save(trainable_state, os.path.join(path, "freeze_tuned.pt"))
        processor.save_pretrained(path)
        metadata["ft_method"] = self.name
        metadata["trained_param_names"] = sorted(trained_names)
        with open(os.path.join(path, "mmit_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoVLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        state = torch.load(
            os.path.join(path, "freeze_tuned.pt"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state, strict=False)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} (Freeze: {adapter_name})"}
        return model, processor, info
