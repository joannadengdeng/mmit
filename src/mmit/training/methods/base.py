"""TrainingMethod ABC — the central interface for all fine-tuning methods.

Each method defines how to prepare a model, compute loss, save/load checkpoints,
and what UI parameters it needs in the dashboard.

Built-in methods:
  - QLoRA, LoRA, DoRA          — parameter-efficient LoRA variants
  - FullFT                      — full fine-tuning (all parameters)
  - FreezeTuning                — train selected modules only
  - L2T                         — instruction-aware loss masking (Zhou et al. 2025)
  - MoReS                       — representation steering (Bi et al. 2025)
  - LoRAInLoRA                  — nested LoRA for continual learning

To add a custom method:
  1. Subclass ``TrainingMethod``
  2. Implement all abstract methods
  3. Register: ``registry.register("training_method", "my-method", MyMethod)``

freeze_patterns
---------------
All methods inherit a universal ``freeze_patterns`` mechanism from the base class.
After a method's ``_prepare_model_impl()`` runs, the base class applies any
``freeze_patterns`` from the config to freeze matching parameters by name.
This allows model-agnostic freezing without hardcoding component names.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TrainingMethod(ABC):
    """Base class for all fine-tuning methods.

    A TrainingMethod encapsulates the complete recipe for fine-tuning a VLM:
    how to load the model, what to freeze/adapt, how to compute loss,
    and how to save/load the result.
    """

    name: str = ""              # registry key: "qlora", "lora", "l2t", ...
    display_name: str = ""      # UI label: "QLoRA", "L2T (Zhou et al. 2025)", ...
    paper_ref: str = ""         # citation, empty for classic methods

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Return default hyperparameters for this method."""

    @abstractmethod
    def ui_params(self) -> List[Dict[str, Any]]:
        """Describe UI parameters for the Gradio dashboard.

        Each dict has keys: name, type ("slider"|"number"|"dropdown"|"checkbox"),
        label, default, and optional min/max/step/choices.
        """

    def requires_quantization(self) -> bool:
        """Whether the base model should be loaded in 4-bit quantization.

        Only QLoRA returns True. All other methods load in bf16/fp16.
        """
        return False

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def prepare_model(
        self,
        model: nn.Module,
        processor: Any,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, str]:
        """Prepare the model for training.

        Calls ``_prepare_model_impl()`` (subclass logic), then applies
        ``freeze_patterns`` from config to freeze matching parameters.

        Parameters
        ----------
        model : nn.Module
            The base VLM loaded from HuggingFace.
        processor :
            The tokenizer / processor.
        config : dict
            Method-specific config (from UI or CLI).

        Returns
        -------
        (prepared_model, info_str)
        """
        model, info = self._prepare_model_impl(model, processor, config)

        # Universal post-processing: apply freeze_patterns
        freeze_patterns = config.get("freeze_patterns", [])
        if freeze_patterns:
            frozen_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad and any(p in name for p in freeze_patterns):
                    param.requires_grad = False
                    frozen_count += 1
            if frozen_count > 0:
                info += f"\nfreeze_patterns matched {frozen_count} params"

        return model, info

    @abstractmethod
    def _prepare_model_impl(
        self,
        model: nn.Module,
        processor: Any,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, str]:
        """Subclass implementation of model preparation.

        This is where PEFT injection, hook registration, freezing, etc. happen.
        """

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def preprocess_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_meta: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Optionally modify labels before loss computation.

        Override for methods that need custom masking (e.g., L2T unmasks
        instruction tokens). Default: return labels unchanged.
        """
        return labels

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        outputs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the training loss.

        Returns
        -------
        (loss, metrics_dict) — scalar loss and optional logging metrics.
        """

    @abstractmethod
    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Return optimizer parameter groups.

        Each dict has "params" (list of Parameters) and optionally "lr".
        """

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        processor: Any,
        path: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save trained weights/adapter to disk."""

    @abstractmethod
    def load_for_inference(
        self,
        path: str,
        base_model_id: str,
        **kwargs,
    ) -> Tuple[nn.Module, Any, Dict[str, str]]:
        """Load a saved checkpoint for inference.

        Returns
        -------
        (model, processor, info_dict)
        """
