"""ReFT / LoReFT: Low-rank Representation Finetuning.

Paper: Wu et al., "ReFT: Representation Finetuning for Language Models",
       NeurIPS 2024
arXiv: 2404.03592

Key idea: Instead of modifying model weights (like LoRA), learn task-specific
interventions on hidden representations.  LoReFT applies a low-rank linear
subspace projection to edit hidden states at specific token positions,
achieving 15-65x more parameter efficiency than LoRA.

Adapted for multimodal VLMs: interventions are applied at all positions
(or optionally only visual token positions) across selected LLM layers.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmit.training.methods.base import TrainingMethod
from mmit.training.losses.ce_loss import CrossEntropyLoss

IGNORE_INDEX = -100
_ce_loss = CrossEntropyLoss()


class LoReFTIntervention(nn.Module):
    """Low-rank linear subspace ReFT intervention.

    Learns to edit a low-rank subspace of the hidden representation:
      h' = h + R^T @ (W @ (R @ h) + b - R @ h)

    where R ∈ R^{r×d} is a learned orthogonal projection into the
    intervention subspace, W ∈ R^{r×r} is a learned transformation,
    and b ∈ R^r is a learned bias.

    This is conceptually similar to MoReS but:
    - Operates on all tokens (not just visual), or on a configurable subset
    - Uses learned R (not Householder), with orthogonality regularisation
    - W is r×r (square), enabling richer transformations within the subspace
    """

    def __init__(self, hidden_dim: int, rank: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # R: projection into intervention subspace [rank, hidden_dim]
        self.R = nn.Parameter(torch.empty(rank, hidden_dim))
        nn.init.orthogonal_(self.R)

        # W: transformation within subspace [rank, rank]
        self.W = nn.Linear(rank, rank, bias=True)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def orthogonal_penalty(self) -> torch.Tensor:
        """Soft orthogonality constraint: ||R·R^T - I||_F^2."""
        RRt = self.R @ self.R.t()
        I = torch.eye(self.rank, device=self.R.device, dtype=self.R.dtype)
        return (RRt - I).pow(2).sum()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply intervention to hidden states.

        h: [batch, seq_len, hidden_dim]
        returns: h' with same shape
        """
        # Project into subspace
        Rh = F.linear(h, self.R)         # [B, seq, rank]
        # Transform within subspace
        Wh = self.W(Rh)                  # [B, seq, rank]
        # Compute subspace delta and project back
        delta_low = self.dropout(Wh - Rh)
        delta = F.linear(delta_low, self.R.t())  # [B, seq, hidden_dim]
        return h + delta


# Reuse helpers from mores.py
from mmit.training.methods.mores import (
    _find_llm_layers,
    _get_hidden_dim,
    _parse_positions,
    _detect_image_token_id,
)


class ReFTMethod(TrainingMethod):
    """LoReFT: Low-rank Representation Finetuning for VLMs.

    Injects lightweight learned interventions on hidden representations
    at selected LLM layers.  15-65x more parameter-efficient than LoRA.
    """

    name = "reft"
    display_name = "ReFT (LoReFT)"
    paper_ref = "Wu et al., NeurIPS 2024"

    def __init__(self):
        self._interventions: List[LoReFTIntervention] = []
        self._hooks: list = []
        self._ortho_coeff: float = 0.01
        self._steer_visual_only: bool = False
        self._image_token_id: Optional[int] = None

    def default_config(self):
        return {
            "intervention_rank": 4,
            "positions": "all",
            "dropout": 0.05,
            "share_weights": False,
            "steer_visual_only": False,
            "ortho_loss_coeff": 0.01,
            "train_projector": False,
        }

    def ui_params(self):
        return [
            {"name": "intervention_rank", "type": "slider",
             "label": "Intervention Rank",
             "default": 4, "min": 1, "max": 32, "step": 1},
            {"name": "positions", "type": "dropdown",
             "label": "Layer Positions",
             "default": "all",
             "choices": ["all", "f4+l5", "f8+l8", "l10", "l5"]},
            {"name": "dropout", "type": "number",
             "label": "Dropout", "default": 0.05},
            {"name": "share_weights", "type": "checkbox",
             "label": "Share Weights Across Layers", "default": False},
            {"name": "steer_visual_only", "type": "checkbox",
             "label": "Intervene Visual Tokens Only", "default": False},
        ]

    def _prepare_model_impl(self, model, processor, config):
        rank = int(config.get("intervention_rank", 4))
        positions_str = config.get("positions", "all")
        dropout = float(config.get("dropout", 0.05))
        share_weights = config.get("share_weights", False)
        self._steer_visual_only = config.get("steer_visual_only", False)
        self._ortho_coeff = float(config.get("ortho_loss_coeff", 0.01))
        self._image_token_id = _detect_image_token_id(model)

        # Freeze entire model
        for p in model.parameters():
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = False

        # Find LLM layers
        layer_list, num_layers = _find_llm_layers(model)
        hidden_dim = _get_hidden_dim(model)
        target_indices = _parse_positions(positions_str, num_layers)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Create interventions
        self._hooks = []
        self._interventions = []

        if share_weights:
            shared_iv = LoReFTIntervention(hidden_dim, rank, dropout).to(device=device, dtype=dtype)
            self._interventions = [shared_iv]
            ivs_per_layer = [shared_iv] * len(target_indices)
        else:
            ivs_per_layer = []
            for _ in target_indices:
                iv = LoReFTIntervention(hidden_dim, rank, dropout).to(device=device, dtype=dtype)
                self._interventions.append(iv)
                ivs_per_layer.append(iv)

        # Attach forward hooks
        method_ref = self

        for idx, iv in zip(target_indices, ivs_per_layer):
            def make_hook(intervention):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    if method_ref._steer_visual_only and method_ref._image_token_id is not None:
                        ids = getattr(model, "_reft_input_ids", None)
                        if ids is not None and ids.shape[1] == h.shape[1]:
                            vis_mask = (ids == method_ref._image_token_id)
                            h_new = h.clone()
                            for b in range(h.size(0)):
                                vis_idx = vis_mask[b].nonzero(as_tuple=True)[0]
                                if vis_idx.numel() > 0:
                                    h_sel = h[b, vis_idx].unsqueeze(0)
                                    h_new[b, vis_idx] = intervention(h_sel).squeeze(0)
                            h = h_new
                        else:
                            h = intervention(h)
                    else:
                        h = intervention(h)

                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook

            handle = layer_list[idx].register_forward_hook(make_hook(iv))
            self._hooks.append(handle)

        # Pre-hook to capture input_ids
        def _capture_ids(module, args, kwargs):
            if "input_ids" in kwargs:
                module._reft_input_ids = kwargs["input_ids"]
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                module._reft_input_ids = args[0]

        pre_h = model.register_forward_pre_hook(_capture_ids, with_kwargs=True)
        self._hooks.append(pre_h)

        # Optionally unfreeze projector
        projector_params = 0
        if config.get("train_projector", False):
            for name, module in model.named_modules():
                if "projector" in name.lower():
                    for p in module.parameters():
                        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                            p.requires_grad = True
                            projector_params += p.numel()

        intervention_params = sum(p.numel() for iv in self._interventions for p in iv.parameters())
        total = intervention_params + projector_params

        info = (
            f"ReFT (LoReFT): {len(target_indices)} layers, rank={rank}\n"
            f"Layers: {target_indices}\n"
            f"Weight sharing: {'yes' if share_weights else 'no'}\n"
            f"Visual-only steering: {'yes' if self._steer_visual_only else 'no'}\n"
            f"Intervention params: {intervention_params:,}\n"
            f"Projector params: {projector_params:,}\n"
            f"Total trainable: {total:,}"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        text_loss, metrics = _ce_loss.compute(model, batch, outputs)

        # Orthogonality penalty on R matrices
        ortho_loss = torch.tensor(0.0, device=text_loss.device, dtype=text_loss.dtype)
        for iv in self._interventions:
            ortho_loss = ortho_loss + iv.orthogonal_penalty()

        if ortho_loss.item() > 0:
            total_loss = text_loss + self._ortho_coeff * ortho_loss
            metrics["ortho_loss"] = ortho_loss.item()
        else:
            total_loss = text_loss

        return total_loss, metrics

    def get_trainable_params(self, model):
        intervention_params = []
        for iv in self._interventions:
            intervention_params.extend(p for p in iv.parameters())

        groups = [{"params": intervention_params}]

        # Projector as separate group with lower lr
        projector_params = []
        for name, p in model.named_parameters():
            if p.requires_grad and "projector" in name.lower():
                projector_params.append(p)
        if projector_params:
            groups.append({"params": projector_params, "lr": 2e-5})

        return groups

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)

        # Save intervention states
        iv_states = {}
        for i, iv in enumerate(self._interventions):
            iv_states[f"intervention_{i}"] = iv.state_dict()
        torch.save(iv_states, os.path.join(path, "reft_interventions.pt"))

        # Save projector weights if unfrozen
        proj_state = {}
        for name, p in model.named_parameters():
            if "projector" in name.lower():
                proj_state[name] = p.data
        if proj_state:
            torch.save(proj_state, os.path.join(path, "projector_state.pt"))

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

        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoVLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

        # Load config
        meta_path = os.path.join(path, "mmit_meta.json")
        config = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                config = json.load(f).get("config", {})

        # Re-prepare model (creates hooks)
        self.prepare_model(model, processor, config)

        # Load intervention weights
        iv_path = os.path.join(path, "reft_interventions.pt")
        if os.path.exists(iv_path):
            states = torch.load(iv_path, map_location="cpu", weights_only=True)
            for i, iv in enumerate(self._interventions):
                key = f"intervention_{i}"
                if key in states:
                    iv.load_state_dict(states[key])

        # Load projector weights
        proj_path = os.path.join(path, "projector_state.pt")
        if os.path.exists(proj_path):
            proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
            model.load_state_dict(proj_state, strict=False)

        model.eval()
        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} (ReFT: {adapter_name})"}
        return model, processor, info
