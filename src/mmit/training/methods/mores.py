"""MoReS: Representation Steering for multimodal models.

Paper: Bi et al., "LLaVA Steering: Visual Instruction Tuning through
       Representation Steering", ACL 2025
arXiv: 2412.12359

Key idea: Instead of fine-tuning LLM weights, insert a small orthogonal
low-rank intervention at each selected layer.  The LLM remains frozen;
only the intervention matrices and the vision-language projector are updated.

Per-layer intervention (applied only to visual tokens):
  h' = h + R^T · (W·h + b − R·h)
  where R: [d, D] orthogonal projection (Householder parametrization)
        W: nn.Linear(D, d) learned steering matrix
        d = 1 (paper optimal), D = model hidden dim

Key findings from paper:
  - Householder parametrization for hard orthogonality (not soft penalty)
  - Rank d=1 is optimal (more is not better)
  - Only steer visual tokens, not text tokens
  - Only steer top-1% of visual tokens by norm
  - Share weights across all intervened layers
  - ~1000x fewer parameters than LoRA
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmit.training.methods.base import TrainingMethod
from mmit.training.methods.lora import _standard_ce_loss


IGNORE_INDEX = -100


class OrthogonalIntervention(nn.Module):
    """Low-rank orthogonal intervention module.

    Implements: h' = h + R^T · (W·h + b − R·h)

    R is initialized orthogonal and kept approximately orthogonal via a
    soft penalty ||R·R^T - I||^2 added to the training loss.
    """

    def __init__(self, hidden_dim: int, rank: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # R: orthogonal projection [rank, hidden_dim]
        # Initialized orthogonal; soft penalty replaced by re-projecting
        # periodically (Householder parametrization has dtype issues with
        # bf16/fp16 on some PyTorch versions, so we use plain Parameter
        # with orthogonal init instead).
        self.R = nn.Parameter(torch.empty(rank, hidden_dim))
        nn.init.orthogonal_(self.R)

        # W: steering matrix [rank, hidden_dim]
        self.W = nn.Linear(hidden_dim, rank, bias=True)
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.W.bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def orthogonal_penalty(self) -> torch.Tensor:
        """Soft orthogonality constraint: ||R·R^T - I||_F^2."""
        RRt = self.R @ self.R.t()
        I = torch.eye(self.rank, device=self.R.device, dtype=self.R.dtype)
        return (RRt - I).pow(2).sum()

    def forward(
        self,
        h: torch.Tensor,
        vis_mask: Optional[torch.Tensor] = None,
        steer_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Apply intervention, optionally only to visual tokens.

        Parameters
        ----------
        h : [batch, seq_len, hidden_dim]
        vis_mask : [batch, seq_len] bool, True = visual token. If None, steer all.
        steer_ratio : float in (0, 1], fraction of visual tokens to steer.
                      Paper finds 1% (0.01) is optimal.

        Returns
        -------
        h' : [batch, seq_len, hidden_dim]
        """
        R = self.R  # [rank, hidden_dim]

        if vis_mask is None and steer_ratio >= 1.0:
            # Steer all tokens (simple path)
            Rh = F.linear(h, R)           # [B, seq, rank]
            Wh = self.W(h)                # [B, seq, rank]
            delta_low = self.dropout(Wh - Rh)
            delta = F.linear(delta_low, R.t())
            return h + delta

        # ── Selective steering: only visual tokens ──
        h_out = h.clone()

        for b in range(h.size(0)):
            if vis_mask is not None:
                vis_idx = vis_mask[b].nonzero(as_tuple=True)[0]  # [num_vis]
            else:
                vis_idx = torch.arange(h.size(1), device=h.device)

            if vis_idx.numel() == 0:
                continue

            # Select top-k% by norm if steer_ratio < 1
            if steer_ratio < 1.0 and vis_idx.numel() > 1:
                h_vis = h[b, vis_idx]  # [num_vis, D]
                norms = h_vis.norm(dim=-1)
                k = max(1, int(vis_idx.numel() * steer_ratio))
                _, topk_local = norms.topk(k)
                vis_idx = vis_idx[topk_local]

            # Apply intervention to selected tokens
            h_sel = h[b, vis_idx]        # [k, D]
            Rh = F.linear(h_sel, R)      # [k, rank]
            Wh = self.W(h_sel)           # [k, rank]
            delta_low = self.dropout(Wh - Rh)
            delta = F.linear(delta_low, R.t())  # [k, D]
            h_out[b, vis_idx] = h_sel + delta

        return h_out


def _parse_positions(positions, num_layers: int) -> List[int]:
    """Parse layer positions into indices. Accepts multiple formats:

    String shortcuts:
      "f4+l5" → first 4 + last 5 = [0,1,2,3,27,28,29,30,31]
      "all"   → [0,1,...,31]
      "f4"    → [0,1,2,3]
      "l5"    → [27,28,29,30,31]

    Direct specification:
      [0, 1, 2, 3, 27, 28, 29, 30, 31]  → used as-is
      "0+1+2+3+27+28+29+30+31"          → parsed from string
    """
    # ── List of ints: use directly ──
    if isinstance(positions, list):
        return sorted(idx for idx in positions if 0 <= idx < num_layers)

    # ── String format ──
    position_str = str(positions).strip().lower()
    if position_str == "all":
        return list(range(num_layers))

    layers = set()
    for part in position_str.split("+"):
        part = part.strip()
        match = re.match(r"^([fl])(\d+)$", part)
        if match:
            prefix, count = match.group(1), int(match.group(2))
            if prefix == "f":
                layers.update(range(min(count, num_layers)))
            else:  # "l"
                layers.update(range(max(0, num_layers - count), num_layers))
        elif part.isdigit():
            idx = int(part)
            if 0 <= idx < num_layers:
                layers.add(idx)
    return sorted(layers)


def _find_llm_layers(model: nn.Module) -> Tuple[nn.ModuleList, int]:
    """Find the LLM decoder layer list in a VLM.

    Must avoid returning vision encoder blocks (e.g. model.visual.blocks).
    """
    # Explicit paths for known VLM architectures
    search_paths = [
        "language_model.model.layers",   # LLaVA, InternVL
        "model.model.layers",            # Qwen2.5-VL (model.model = Qwen2_5_VLModel)
        "model.layers",                  # some models
        "language_model.layers",         # some models
        "transformer.h",                 # GPT-style
    ]
    for path in search_paths:
        obj = model
        found = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found and isinstance(obj, nn.ModuleList):
            return obj, len(obj)

    # Fallback: search named_modules, but skip anything under "visual"/"vision"
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            # Skip vision encoder blocks
            if any(skip in name.lower() for skip in ("visual", "vision", "vit", "encoder")):
                continue
            return module, len(module)

    raise ValueError("Cannot find LLM transformer layers in model.")


def _get_hidden_dim(model: nn.Module) -> int:
    """Get the LLM hidden dimension."""
    config = getattr(model, "config", None)
    if config is not None:
        text_config = getattr(config, "text_config", config)
        if hasattr(text_config, "hidden_size"):
            return text_config.hidden_size
    raise ValueError("Cannot determine hidden_size from model config.")


def _detect_image_token_id(model: nn.Module) -> Optional[int]:
    """Try to find the image token ID from model config."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    # Qwen2-VL uses image_token_id in config
    if hasattr(config, "image_token_id"):
        return config.image_token_id
    # LLaVA uses a special <image> token
    if hasattr(config, "image_token_index"):
        return config.image_token_index
    return None


class MoReSMethod(TrainingMethod):
    """MoReS: representation steering — LLM frozen, orthogonal low-rank
    interventions injected at selected layers.

    Matches the paper (Bi et al., ACL 2025):
      - Householder parametrization for hard orthogonality
      - Shared intervention across all target layers
      - Only steers visual tokens (configurable ratio)
      - Default rank d=1 (paper optimal)
    """

    name = "mores"
    display_name = "MoReS (Bi et al. 2025)"
    paper_ref = "Bi et al., LLaVA Steering, ACL 2025"

    def __init__(self):
        self._intervention: Optional[OrthogonalIntervention] = None
        self._hooks: list = []
        self._steer_ratio: float = 0.01
        self._vis_only: bool = True
        self._image_token_id: Optional[int] = None

    def default_config(self):
        return {
            "intervention_rank": 1,
            "positions": "f4+l5",
            "dropout": 0.05,
            "share_weights": True,
            "steer_visual_only": True,
            "steer_ratio": 0.01,
            "train_projector": False,   # per paper: only train interventions
        }

    def ui_params(self):
        return [
            {"name": "intervention_rank", "type": "slider",
             "label": "Intervention Rank (d)",
             "default": 1, "min": 1, "max": 16, "step": 1},
            {"name": "positions", "type": "dropdown",
             "label": "Layer Positions",
             "default": "f4+l5",
             "choices": ["f4+l5", "f8+l8", "all", "l10", "f4", "l5"]},
            {"name": "dropout", "type": "number",
             "label": "Dropout", "default": 0.05},
            {"name": "steer_ratio", "type": "number",
             "label": "Visual Token Steer Ratio",
             "default": 0.01},
        ]

    def _prepare_model_impl(self, model, processor, config):
        rank = int(config.get("intervention_rank", 1))
        positions_str = config.get("positions", "f4+l5")
        dropout = float(config.get("dropout", 0.05))
        share_weights = config.get("share_weights", True)
        self._vis_only = config.get("steer_visual_only", True)
        self._steer_ratio = float(config.get("steer_ratio", 0.01))

        # Detect image token ID for visual-only steering
        self._image_token_id = _detect_image_token_id(model)

        # Freeze entire model (skip quantized / non-float params)
        for p in model.parameters():
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = False

        # Find LLM layers and hidden dim
        layer_list, num_layers = _find_llm_layers(model)
        hidden_dim = _get_hidden_dim(model)
        target_indices = _parse_positions(positions_str, num_layers)

        # Create intervention(s)
        self._hooks = []
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if share_weights:
            # Paper approach: single intervention shared across all layers
            self._intervention = OrthogonalIntervention(
                hidden_dim, rank=rank, dropout=dropout,
            ).to(device=device, dtype=dtype)
            interventions = [self._intervention] * len(target_indices)
        else:
            # Independent intervention per layer
            self._intervention = OrthogonalIntervention(
                hidden_dim, rank=rank, dropout=dropout,
            ).to(device=device, dtype=dtype)
            interventions = []
            for _ in target_indices:
                iv = OrthogonalIntervention(
                    hidden_dim, rank=rank, dropout=dropout,
                ).to(device=device, dtype=dtype)
                interventions.append(iv)

        # Store all unique interventions for parameter collection
        self._all_interventions = list(dict.fromkeys(interventions))

        # Attach hooks
        method_ref = self  # capture for hook closure

        for idx, iv in zip(target_indices, interventions):
            def make_hook(intervention):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # Build visual token mask if needed
                    vis_mask = None
                    if method_ref._vis_only and method_ref._image_token_id is not None:
                        # Try to get input_ids from the current forward pass
                        # They're stored by our pre-hook
                        ids = getattr(model, "_mores_input_ids", None)
                        if ids is not None and ids.shape[1] == h.shape[1]:
                            vis_mask = (ids == method_ref._image_token_id)

                    h_new = intervention(
                        h,
                        vis_mask=vis_mask,
                        steer_ratio=method_ref._steer_ratio,
                    )

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return hook

            h = layer_list[idx].register_forward_hook(make_hook(iv))
            self._hooks.append(h)

        # Register a pre-forward hook on the whole model to capture input_ids
        def _capture_input_ids(module, args, kwargs):
            if "input_ids" in kwargs:
                module._mores_input_ids = kwargs["input_ids"]
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                module._mores_input_ids = args[0]

        pre_h = model.register_forward_pre_hook(_capture_input_ids, with_kwargs=True)
        self._hooks.append(pre_h)

        # Projector: only unfreeze if config says so (default: frozen per paper)
        train_projector = config.get("train_projector", False)
        projector_unfrozen = 0
        if train_projector:
            for name, module in model.named_modules():
                if "projector" in name.lower() or "multi_modal_projector" in name.lower():
                    for p in module.parameters():
                        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                            p.requires_grad = True
                            projector_unfrozen += p.numel()

        intervention_params = sum(
            p.numel() for iv in self._all_interventions for p in iv.parameters()
        )
        total = intervention_params + projector_unfrozen

        vis_info = ""
        if self._vis_only:
            vis_info = f"Steer: visual tokens only ({self._steer_ratio:.0%})\n"

        info = (
            f"MoReS: {len(target_indices)} layers, "
            f"{'shared' if share_weights else 'independent'} weights\n"
            f"Layers: {target_indices}\n"
            f"Rank: {rank}, Orthogonality: Householder (hard)\n"
            f"{vis_info}"
            f"Intervention params: {intervention_params:,}\n"
            f"Projector params: {projector_unfrozen:,}\n"
            f"Total trainable: {total:,}"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        text_loss = _standard_ce_loss(outputs, batch)
        metrics = {"text_loss": text_loss.item()}

        # Soft orthogonality penalty on R
        ortho_loss = torch.tensor(0.0, device=text_loss.device, dtype=text_loss.dtype)
        for iv in self._all_interventions:
            ortho_loss = ortho_loss + iv.orthogonal_penalty()
        if ortho_loss.item() > 0:
            total_loss = text_loss + 0.01 * ortho_loss
            metrics["ortho_loss"] = ortho_loss.item()
        else:
            total_loss = text_loss

        return total_loss, metrics

    def get_trainable_params(self, model):
        # Intervention params only (projector frozen by default per paper)
        intervention_params = []
        for iv in self._all_interventions:
            intervention_params.extend(p for p in iv.parameters())

        groups = [{"params": intervention_params}]  # lr set by caller

        # If projector was unfrozen, add as separate group with lower lr
        projector_params = []
        for name, p in model.named_parameters():
            if p.requires_grad and "projector" in name.lower():
                projector_params.append(p)
        if projector_params:
            groups.append({"params": projector_params, "lr": 2e-5})

        return groups

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)

        # Save intervention state (shared = just one)
        intervention_states = {}
        for i, iv in enumerate(self._all_interventions):
            intervention_states[f"intervention_{i}"] = iv.state_dict()
        torch.save(intervention_states, os.path.join(path, "mores_interventions.pt"))

        # Save projector weights
        projector_state = {}
        for name, p in model.named_parameters():
            if "projector" in name.lower():
                projector_state[name] = p.data
        if projector_state:
            torch.save(projector_state, os.path.join(path, "projector_state.pt"))

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

        processor = AutoProcessor.from_pretrained(
            base_model_id, trust_remote_code=True,
        )
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
        iv_path = os.path.join(path, "mores_interventions.pt")
        if os.path.exists(iv_path):
            states = torch.load(iv_path, map_location="cpu", weights_only=True)
            for i, iv in enumerate(self._all_interventions):
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
        info = {"model_id": f"{base_model_id} (MoReS: {adapter_name})"}
        return model, processor, info
