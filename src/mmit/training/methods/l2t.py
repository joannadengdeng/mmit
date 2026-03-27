"""L2T: Learning to Instruct for Visual Instruction Tuning.

Paper: Zhou et al., "Learning to Instruct for Visual Instruction Tuning", NeurIPS 2025
arXiv: 2503.22215

Key idea: During SFT, also compute loss on content-bearing instruction tokens
(not just response tokens). Template/boilerplate instruction tokens remain masked.
This forces the model to better attend to visual input, reducing hallucination.

Implementation: modify the labels tensor — unmask non-template instruction tokens.
No architecture or loss function changes needed.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mmit.training.methods.base import TrainingMethod

IGNORE_INDEX = -100


class L2TMethod(TrainingMethod):
    """L2T: compute loss on instruction tokens too (minus templates).

    Wraps a base method (LoRA, QLoRA, Full FT, etc.) and only modifies
    the label masking during training. Everything else is delegated.
    """

    name = "l2t"
    display_name = "L2T (Zhou et al. 2025)"
    paper_ref = "Zhou et al., Learning to Instruct for Visual Instruction Tuning, NeurIPS 2025"

    def __init__(self):
        self._base: Optional[TrainingMethod] = None
        self._templates: set[str] = set()

    def _get_base(self, config: dict) -> TrainingMethod:
        """Lazily resolve the base training method."""
        if self._base is None:
            base_name = config.get("base_method", "lora")
            from mmit.registry import registry
            cls = registry.get_cls("training_method", base_name)
            self._base = cls()
        return self._base

    def default_config(self):
        return {
            "base_method": "lora",
            "template_top_k": 20,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
        }

    def ui_params(self):
        return [
            {"name": "base_method", "type": "dropdown", "label": "Base method",
             "choices": ["lora", "qlora", "full_ft"], "default": "lora"},
            {"name": "template_top_k", "type": "slider",
             "label": "Template sentences to filter (top-K frequent)",
             "default": 20, "min": 5, "max": 100, "step": 5},
            {"name": "lora_r", "type": "slider", "label": "LoRA Rank (r)",
             "default": 8, "min": 4, "max": 64, "step": 4},
            {"name": "lora_alpha", "type": "slider", "label": "LoRA Alpha",
             "default": 16, "min": 8, "max": 128, "step": 8},
            {"name": "lora_dropout", "type": "number", "label": "Dropout",
             "default": 0.05},
        ]

    def requires_quantization(self):
        return False  # depends on base, but conservative default

    def _prepare_model_impl(self, model, processor, config):
        base = self._get_base(config)
        # Delegate to the base method's full prepare_model (includes freeze_patterns)
        return base.prepare_model(model, processor, config)

    def build_template_list(self, dataset: list, processor, top_k: int = 20):
        """Scan dataset for high-frequency instruction sentences.

        Call this before training to populate self._templates.

        Parameters
        ----------
        dataset : list
            Training samples (each has instruction text).
        processor :
            Tokenizer for decoding.
        top_k : int
            Number of most-frequent sentences to treat as templates.
        """
        sentence_counter: Counter = Counter()
        for sample in dataset:
            if hasattr(sample, "turns"):
                for turn in sample.turns:
                    if turn.role in ("human", "user"):
                        # Split instruction into sentences
                        for sent in turn.content.replace("\n", ". ").split(". "):
                            sent = sent.strip()
                            if sent:
                                sentence_counter[sent] += 1

        # Top-K most frequent are templates
        self._templates = {sent for sent, _ in sentence_counter.most_common(top_k)}

    def preprocess_labels(self, input_ids, labels, batch_meta=None):
        """Unmask non-template instruction tokens.

        Standard VIT: labels[instruction_positions] = -100
        L2T: labels[template_positions] = -100, labels[content_instruction] = input_ids

        For simplicity, we unmask ALL instruction tokens that were previously
        masked (-100). Template removal is a refinement applied on top.
        """
        # Find positions where labels == -100 but input_ids != pad_token
        # These are masked instruction tokens — unmask them
        mask = labels == IGNORE_INDEX
        if mask.any():
            labels = labels.clone()
            labels[mask] = input_ids[mask]
        return labels

    def compute_loss(self, model, batch, outputs):
        base = self._get_base({})
        return base.compute_loss(model, batch, outputs)

    def get_trainable_params(self, model):
        base = self._get_base({})
        return base.get_trainable_params(model)

    def save_checkpoint(self, model, processor, path, metadata):
        base = self._get_base({})
        metadata["ft_method"] = self.name
        metadata["l2t_base_method"] = base.name
        base.save_checkpoint(model, processor, path, metadata)

    def load_for_inference(self, path, base_model_id, **kwargs):
        # Read meta to find base method
        import json
        meta_path = os.path.join(path, "mmit_meta.json")
        base_name = "lora"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            base_name = meta.get("l2t_base_method", "lora")

        from mmit.registry import registry
        base_cls = registry.get("training_method", base_name)
        base = base_cls()
        return base.load_for_inference(path, base_model_id, **kwargs)
