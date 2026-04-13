"""LLaVA-MoLE: Sparse Mixture of LoRA Experts.

Paper: Chen et al., "LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating
       Data Conflicts in Instruction Finetuning MLLMs", 2024
arXiv: 2401.16160

Key idea: Replace a single LoRA adapter in each MLP layer with multiple LoRA
experts and a lightweight router.  The router selects top-1 expert per token,
mitigating data conflicts when mixing diverse instruction datasets (e.g. OCR +
VQA + science).  Training cost stays roughly constant due to sparse activation.

Implementation notes:
  - LoRA experts are injected into MLP gate/up/down projections only
  - Attention projections use a single standard LoRA (no MoE)
  - Router is a simple linear layer with softmax
  - Auxiliary load-balancing loss encourages uniform expert usage
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


# ── MoLE Modules ─────────────────────────────────────────────────────

class LoRAExpert(nn.Module):
    """A single LoRA expert: down projection → up projection."""

    def __init__(self, in_features: int, out_features: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Kaiming init for A, zero init for B (standard LoRA)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x)))


class MoLELayer(nn.Module):
    """Mixture-of-LoRA-Experts layer wrapping a frozen linear module.

    Routes each token to the top-1 expert, applies the selected LoRA delta,
    and adds it to the frozen module's output.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.num_experts = num_experts
        self.scaling = alpha / rank

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        # Freeze the base linear
        for p in base_linear.parameters():
            p.requires_grad = False

        # Experts
        self.experts = nn.ModuleList([
            LoRAExpert(in_f, out_f, rank, dropout) for _ in range(num_experts)
        ])

        # Router: simple linear → softmax
        self.router = nn.Linear(in_f, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)

        # Tracking for load-balancing loss
        self._routing_probs: Optional[torch.Tensor] = None
        self._expert_mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen)
        base_out = self.base_linear(x)

        # Route: [*, num_experts]
        router_logits = self.router(x.detach())  # detach input to router (paper)
        routing_probs = F.softmax(router_logits, dim=-1)
        self._routing_probs = routing_probs

        # Top-1 selection
        expert_idx = routing_probs.argmax(dim=-1)  # [*]
        self._expert_mask = F.one_hot(expert_idx, self.num_experts).float()

        # Dispatch to experts
        # For efficiency, process all tokens per expert in a batch
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # [N, in_f]
        idx_flat = expert_idx.reshape(-1)     # [N]
        delta = torch.zeros(
            x_flat.shape[0], base_out.shape[-1],
            device=x.device, dtype=x.dtype,
        )

        for i, expert in enumerate(self.experts):
            mask = (idx_flat == i)
            if mask.any():
                delta[mask] = expert(x_flat[mask])

        delta = delta.reshape(*orig_shape, -1)
        return base_out + delta * self.scaling


def _inject_mole_layers(
    model: nn.Module,
    num_experts: int,
    rank: int,
    alpha: float,
    dropout: float,
    freeze_patterns: List[str],
) -> Tuple[List[MoLELayer], List[str]]:
    """Inject MoLE layers into MLP projections of the LLM.

    Returns the list of MoLE layers (for parameter collection) and info lines.
    """
    mole_layers: List[MoLELayer] = []
    mlp_targets = {"gate_proj", "up_proj", "down_proj"}

    for name, module in list(model.named_modules()):
        # Skip frozen patterns (e.g. vision encoder)
        if freeze_patterns and any(p in name for p in freeze_patterns):
            continue

        short = name.split(".")[-1]
        if short in mlp_targets and isinstance(module, nn.Linear):
            mole = MoLELayer(module, num_experts, rank, alpha, dropout)
            # Move experts + router to same device/dtype as the base linear
            device = module.weight.device
            dtype = module.weight.dtype
            mole.experts.to(device=device, dtype=dtype)
            mole.router.to(device=device, dtype=dtype)
            # Replace the module in the parent
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], mole)
            else:
                setattr(model, name, mole)
            mole_layers.append(mole)

    return mole_layers, [f"Injected {len(mole_layers)} MoLE layers"]


# Also inject standard single LoRA into attention (non-MoE)
def _inject_attn_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    freeze_patterns: List[str],
) -> int:
    """Inject standard LoRA (via PEFT) into attention projections."""
    from peft import LoraConfig, get_peft_model, TaskType

    attn_targets = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Filter targets to only those actually present (and not frozen)
    found = set()
    for name, _ in model.named_modules():
        if freeze_patterns and any(p in name for p in freeze_patterns):
            continue
        short = name.split(".")[-1]
        if short in attn_targets:
            found.add(short)

    if not found:
        return 0

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=sorted(found),
        task_type=TaskType.CAUSAL_LM,
    )
    get_peft_model(model, lora_config)
    return len(found)


class MoLEMethod(TrainingMethod):
    """LLaVA-MoLE: Sparse Mixture of LoRA Experts.

    MLP layers get multiple LoRA experts with a router (top-1 sparse);
    attention layers get standard single LoRA.
    """

    name = "mole"
    display_name = "LLaVA-MoLE"
    paper_ref = "Chen et al., 2024"

    def __init__(self):
        self._mole_layers: List[MoLELayer] = []
        self._balance_coeff: float = 0.01

    def default_config(self):
        return {
            "num_experts": 4,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "balance_loss_coeff": 0.01,
            "freeze_patterns": ["visual", "vision"],
        }

    def ui_params(self):
        return [
            {"name": "num_experts", "type": "slider",
             "label": "Number of Experts", "default": 4, "min": 2, "max": 8, "step": 1},
            {"name": "lora_r", "type": "slider",
             "label": "LoRA Rank (r)", "default": 8, "min": 4, "max": 64, "step": 4},
            {"name": "lora_alpha", "type": "slider",
             "label": "LoRA Alpha", "default": 16, "min": 8, "max": 128, "step": 8},
            {"name": "lora_dropout", "type": "number",
             "label": "Dropout", "default": 0.05},
            {"name": "balance_loss_coeff", "type": "number",
             "label": "Load Balance Loss Coeff", "default": 0.01},
        ]

    def _prepare_model_impl(self, model, processor, config):
        num_experts = int(config.get("num_experts", 4))
        rank = int(config.get("lora_r", 8))
        alpha = float(config.get("lora_alpha", 16))
        dropout = float(config.get("lora_dropout", 0.05))
        freeze_patterns = config.get("freeze_patterns", ["visual", "vision"])
        self._balance_coeff = float(config.get("balance_loss_coeff", 0.01))

        # Step 1: Freeze the entire model
        for p in model.parameters():
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = False

        # Step 2: Inject MoLE layers into MLP
        self._mole_layers, info_lines = _inject_mole_layers(
            model, num_experts, rank, alpha, dropout, freeze_patterns,
        )

        # Step 3: Inject standard LoRA into attention via PEFT
        attn_count = _inject_attn_lora(model, rank, alpha, dropout, freeze_patterns)

        # Count params
        mole_params = sum(
            p.numel() for layer in self._mole_layers
            for p in layer.parameters() if p.requires_grad
        )
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        info = (
            f"LLaVA-MoLE: {num_experts} experts, rank={rank}, alpha={alpha}\n"
            f"MoLE MLP layers: {len(self._mole_layers)}\n"
            f"Attention LoRA targets: {attn_count}\n"
            f"MoLE params: {mole_params:,}\n"
            f"Total trainable: {total_trainable:,} / {total:,} "
            f"({100*total_trainable/total:.2f}%)"
        )
        return model, info

    def _load_balance_loss(self) -> torch.Tensor:
        """Auxiliary load-balancing loss across all MoLE layers.

        Encourages uniform expert usage:
          L_balance = N * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i,
              P_i = mean routing probability for expert i.
        """
        total_loss = 0.0
        count = 0
        for layer in self._mole_layers:
            if layer._routing_probs is None or layer._expert_mask is None:
                continue
            # routing_probs: [*, num_experts], expert_mask: [*, num_experts]
            probs = layer._routing_probs.reshape(-1, layer.num_experts)
            mask = layer._expert_mask.reshape(-1, layer.num_experts)
            # f_i: fraction routed to each expert
            f = mask.mean(dim=0)
            # P_i: mean probability for each expert
            P = probs.mean(dim=0)
            total_loss = total_loss + (f * P).sum() * layer.num_experts
            count += 1
        if count == 0:
            return torch.tensor(0.0)
        return total_loss / count

    def compute_loss(self, model, batch, outputs):
        text_loss, metrics = _ce_loss.compute(model, batch, outputs)

        balance_loss = self._load_balance_loss()
        if isinstance(balance_loss, torch.Tensor) and balance_loss.item() > 0:
            total_loss = text_loss + self._balance_coeff * balance_loss
            metrics["balance_loss"] = balance_loss.item()
        else:
            total_loss = text_loss

        return total_loss, metrics

    def get_trainable_params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)

        # Save MoLE layer states (experts + routers)
        mole_state = {}
        for i, layer in enumerate(self._mole_layers):
            mole_state[f"mole_layer_{i}"] = {
                "experts": layer.experts.state_dict(),
                "router": layer.router.state_dict(),
            }
        torch.save(mole_state, os.path.join(path, "mole_experts.pt"))

        # Save PEFT adapter (attention LoRA) if present
        try:
            model.save_pretrained(path)
        except AttributeError:
            pass

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

        # Re-prepare model (injects MoLE layers + attention LoRA)
        self.prepare_model(model, processor, config)

        # Load MoLE expert weights
        mole_path = os.path.join(path, "mole_experts.pt")
        if os.path.exists(mole_path):
            mole_state = torch.load(mole_path, map_location="cpu", weights_only=True)
            for i, layer in enumerate(self._mole_layers):
                key = f"mole_layer_{i}"
                if key in mole_state:
                    layer.experts.load_state_dict(mole_state[key]["experts"])
                    layer.router.load_state_dict(mole_state[key]["router"])

        # Load attention LoRA adapter
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()
        except Exception:
            pass

        model.eval()
        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} (MoLE: {adapter_name})"}
        return model, processor, info
