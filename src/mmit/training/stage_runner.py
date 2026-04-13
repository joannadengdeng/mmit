"""StageRunner — orchestrates multi-stage training pipelines.

Each stage is a self-contained training configuration with its own:
  - Data sources (mixer + filter)
  - Preprocessor
  - Training method
  - Loss function
  - Hyperparameters (lr, epochs, etc.)

Stages run sequentially. The model is passed in-memory between stages.
Checkpoints are saved per-stage and at the end.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

IGNORE_INDEX = -100


def _emit(event_type: str, data: dict):
    """Print a JSON event line to stdout."""
    print(json.dumps({"type": event_type, "data": data}), flush=True)


def _cosine_schedule(optimizer, num_warmup: int, num_total: int):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


@dataclass
class StageConfig:
    """Configuration for a single training stage."""
    name: str = ""
    # Data
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    mixer: str = "concat"
    filter_config: Optional[Dict[str, Any]] = None
    # Preprocessor
    preprocessor: str = "chat_template"
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    # Method
    training_method: str = "qlora"
    method_params: Dict[str, Any] = field(default_factory=dict)
    # Loss
    loss: str = "ce"
    loss_params: Dict[str, Any] = field(default_factory=dict)
    # Training hyperparams
    num_epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_steps: int = 500
    output_dir: str = "output"
    # Checkpoint chaining
    resume_from: str = ""


class StageRunner:
    """Orchestrates multi-stage training."""

    def __init__(self, model_path: str, model_family: str = "",
                 experiment_tracker=None):
        self.model_path = model_path
        self.model_family = model_family
        self._model = None
        self._processor = None
        self._tracker = experiment_tracker  # Optional ExperimentTracker

    def _load_model(self, method_obj):
        """Load base model and processor."""
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        _emit("log", {"message": f"Loading model: {self.model_path}", "level": "INFO"})
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
        )

        if method_obj.requires_quantization():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoVLM.from_pretrained(
                self.model_path, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True,
            )
        else:
            self._model = AutoVLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )

    def _build_dataset(self, stage: StageConfig):
        """Build dataset from stage config using mixer and filter."""
        from mmit.registry import registry
        from mmit.training.data.mixer import DataSource

        sources = []
        for src_cfg in stage.data_sources:
            adapter = registry.build(
                "dataset", src_cfg.get("adapter", "hf_datasets"),
                **{k: v for k, v in src_cfg.items()
                   if k not in ("adapter", "weight", "instruction_suffix", "max_samples")},
            )
            sources.append(DataSource(
                adapter=adapter,
                weight=src_cfg.get("weight", 1.0),
                instruction_suffix=src_cfg.get("instruction_suffix", ""),
                max_samples=src_cfg.get("max_samples", 0),
            ))

        mixer = registry.build("mixer", stage.mixer)
        samples = mixer.mix(sources)

        # Apply filter if configured
        if stage.filter_config:
            filter_type = stage.filter_config.get("type", "composite")
            filter_obj = registry.build(
                "filter", filter_type, **stage.filter_config.get("params", {}),
            )
            samples = [s for s in samples if filter_obj.filter(s)]

        return samples

    def _preprocess_dataset(self, samples, stage: StageConfig):
        """Tokenize all samples using the configured preprocessor."""
        from mmit.registry import registry

        preprocessor = registry.build(
            "preprocessor", stage.preprocessor, **stage.preprocessor_params,
        )
        processed = []
        for s in samples:
            try:
                tokenized = preprocessor.tokenize(
                    s, self._processor,
                    image_root=stage.data_sources[0].get("image_root", "") if stage.data_sources else "",
                    max_length=stage.preprocessor_params.get("max_length", 2048),
                )
                processed.append(tokenized)
            except Exception as e:
                _emit("log", {"message": f"Skipping sample {s.id}: {e}", "level": "WARNING"})
        return processed, preprocessor

    def run_stage(self, stage: StageConfig):
        """Run a single training stage."""
        _emit("status", {"status": "loading", "stage": stage.name})

        # Resolve method
        from mmit.registry import registry
        method_obj = registry.build("training_method", stage.training_method)

        # Load model if not already loaded
        if self._model is None:
            self._load_model(method_obj)

        # Build dataset
        _emit("log", {"message": f"[{stage.name}] Loading dataset...", "level": "INFO"})
        samples = self._build_dataset(stage)
        _emit("log", {"message": f"[{stage.name}] {len(samples)} samples", "level": "INFO"})

        # Preprocess
        _emit("log", {"message": f"[{stage.name}] Preprocessing...", "level": "INFO"})
        processed, preprocessor = self._preprocess_dataset(samples, stage)
        _emit("log", {"message": f"[{stage.name}] {len(processed)} samples tokenized", "level": "INFO"})

        if not processed:
            _emit("error", {"message": f"[{stage.name}] No samples after preprocessing"})
            return

        # Prepare model
        ft_config = {**method_obj.default_config(), **stage.method_params}
        self._model, info_str = method_obj.prepare_model(
            self._model, self._processor, ft_config,
        )
        _emit("log", {"message": info_str, "level": "INFO"})

        # Apply loss on_prepare hook
        loss_obj = registry.build("loss", stage.loss, **stage.loss_params)
        loss_obj.on_prepare(self._model, stage.loss_params)

        # Training setup
        param_groups = method_obj.get_trainable_params(self._model)
        for pg in param_groups:
            pg.setdefault("lr", stage.learning_rate)
        optimizer = AdamW(param_groups, weight_decay=stage.weight_decay)

        loader = DataLoader(
            processed,
            batch_size=stage.per_device_batch_size,
            shuffle=True,
            collate_fn=preprocessor.collate,
            drop_last=True,
        )

        steps_per_epoch = max(1, len(loader) // stage.gradient_accumulation_steps)
        total_steps = steps_per_epoch * stage.num_epochs
        warmup_steps = int(total_steps * stage.warmup_ratio)
        scheduler = _cosine_schedule(optimizer, warmup_steps, total_steps)

        # Training loop
        _emit("status", {"status": "training", "stage": stage.name})
        self._model.train()
        device = next(self._model.parameters()).device
        global_step = 0
        total_loss = 0.0
        start_time = time.time()

        for epoch in range(stage.num_epochs):
            for step, batch in enumerate(loader):
                batch = _to_device(batch, device)

                batch["labels"] = method_obj.preprocess_labels(
                    batch["input_ids"], batch["labels"],
                )

                outputs = self._model(**batch)
                loss, metrics = loss_obj.compute(self._model, batch, outputs)
                loss = loss / stage.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % stage.gradient_accumulation_steps == 0:
                    if stage.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for pg in param_groups for p in pg["params"]],
                            stage.max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    total_loss += loss.item() * stage.gradient_accumulation_steps

                    elapsed = time.time() - start_time
                    eta = elapsed / global_step * (total_steps - global_step) if global_step > 0 else 0
                    step_metrics = {
                        "stage": stage.name,
                        "step": global_step,
                        "total": total_steps,
                        "epoch": epoch,
                        "total_epochs": stage.num_epochs,
                        "loss": round(loss.item() * stage.gradient_accumulation_steps, 6),
                        "avg_loss": round(total_loss / global_step, 6),
                        "lr": scheduler.get_last_lr()[0],
                        "eta": round(eta, 1),
                        **{k: round(v, 6) for k, v in metrics.items()},
                    }
                    _emit("metric", step_metrics)

                    # Persist to experiment tracker (if available)
                    if self._tracker is not None:
                        self._tracker.log_train_step(**{
                            k: v for k, v in step_metrics.items()
                            if k not in ("stage", "total", "total_epochs")
                        })

                    if stage.save_steps > 0 and global_step % stage.save_steps == 0:
                        ckpt_path = os.path.join(stage.output_dir, f"checkpoint-{global_step}")
                        method_obj.save_checkpoint(self._model, self._processor, ckpt_path, {
                            "base_model": self.model_path,
                            "stage": stage.name,
                            "step": global_step,
                        })

        # Save final checkpoint for this stage
        final_path = os.path.join(stage.output_dir, "final")
        # If tracker available, save checkpoint into experiment dir
        if self._tracker is not None:
            final_path = self._tracker.get_checkpoint_dir()
        method_obj.save_checkpoint(self._model, self._processor, final_path, {
            "base_model": self.model_path,
            "stage": stage.name,
            "final_loss": round(total_loss / max(1, global_step), 6),
        })

        # Record training summary to experiment tracker
        if self._tracker is not None:
            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self._model.parameters())
            self._tracker.log_train_summary(
                avg_loss=total_loss / max(1, global_step),
                total_steps=global_step,
                train_time_s=time.time() - start_time,
                trainable_params=trainable,
                total_params=total_params,
            )
            self._tracker.set_checkpoint_path(final_path)

        _emit("status", {
            "status": "stage_completed",
            "stage": stage.name,
            "result": f"{global_step} steps, avg loss={total_loss / max(1, global_step):.4f}",
        })

    def run(self, stages: List[StageConfig]):
        """Run all stages sequentially."""
        for i, stage in enumerate(stages):
            _emit("log", {
                "message": f"=== Stage {i+1}/{len(stages)}: {stage.name} ===",
                "level": "INFO",
            })
            self.run_stage(stage)
        _emit("status", {"status": "completed"})
