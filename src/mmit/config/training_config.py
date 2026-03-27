"""Training configuration: YAML loading, validation, and conversion.

Loads a YAML config file and produces the dict structure expected by
``mmit.training.__main__.main()``.

Usage::

    from mmit.config.training_config import load_config, config_to_trainer_dict

    cfg = load_config("configs/local_qlora.yaml")
    trainer_dict = config_to_trainer_dict(cfg)
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class SSHConfig:
    host: str = ""
    port: int = 22
    username: str = ""
    key_path: str = ""
    password: str = ""
    conda_env: str = ""


@dataclass
class ColabConfig:
    mount_drive: bool = True
    drive_mount_point: str = "/content/drive"
    pip_install: List[str] = field(default_factory=list)
    output_to_drive: bool = True


@dataclass
class RuntimeConfig:
    mode: str = "local"  # "local" | "colab" | "ssh"
    ssh: SSHConfig = field(default_factory=SSHConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)


@dataclass
class ModelConfig:
    model_path: str = ""
    family: str = ""  # auto-detected if empty


@dataclass
class TrainingParams:
    ft_method: str = "qlora"
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_steps: int = 500
    output_dir: str = "output"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    adapter: str = "hf_datasets"
    data_path: str = ""
    split: str = "train"
    image_root: str = ""
    max_samples: int = 0


@dataclass
class TrainingConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    data: DataConfig = field(default_factory=DataConfig)


# ── YAML loading ─────────────────────────────────────────────────────

def _parse_ssh(raw: dict) -> SSHConfig:
    if not raw:
        return SSHConfig()
    return SSHConfig(
        host=str(raw.get("host", "")),
        port=int(raw.get("port", 22)),
        username=str(raw.get("username", "")),
        key_path=str(raw.get("key_path", "")),
        password=str(raw.get("password", "")),
        conda_env=str(raw.get("conda_env", "")),
    )


def _parse_colab(raw: dict) -> ColabConfig:
    if not raw:
        return ColabConfig()
    return ColabConfig(
        mount_drive=bool(raw.get("mount_drive", True)),
        drive_mount_point=str(raw.get("drive_mount_point", "/content/drive")),
        pip_install=list(raw.get("pip_install", [])),
        output_to_drive=bool(raw.get("output_to_drive", True)),
    )


def load_config(path: str) -> TrainingConfig:
    """Load and validate a YAML training config file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    TrainingConfig
        Validated config with method defaults merged into ``training.params``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # ── Parse sections ──
    raw_runtime = raw.get("runtime", {})
    raw_model = raw.get("model", {})
    raw_training = raw.get("training", {})
    raw_data = raw.get("data", {})

    cfg = TrainingConfig(
        runtime=RuntimeConfig(
            mode=str(raw_runtime.get("mode", "local")),
            ssh=_parse_ssh(raw_runtime.get("ssh")),
            colab=_parse_colab(raw_runtime.get("colab")),
        ),
        model=ModelConfig(
            model_path=str(raw_model.get("model_path", "")),
            family=str(raw_model.get("family", "")),
        ),
        training=TrainingParams(
            ft_method=str(raw_training.get("ft_method", "qlora")),
            num_epochs=int(raw_training.get("num_epochs", 3)),
            per_device_batch_size=int(raw_training.get("per_device_batch_size", 4)),
            gradient_accumulation_steps=int(raw_training.get("gradient_accumulation_steps", 4)),
            learning_rate=float(raw_training.get("learning_rate", 2e-4)),
            warmup_ratio=float(raw_training.get("warmup_ratio", 0.03)),
            weight_decay=float(raw_training.get("weight_decay", 0.0)),
            max_grad_norm=float(raw_training.get("max_grad_norm", 1.0)),
            save_steps=int(raw_training.get("save_steps", 500)),
            output_dir=str(raw_training.get("output_dir", "output")),
            params=dict(raw_training.get("params", {})),
        ),
        data=DataConfig(
            adapter=str(raw_data.get("adapter", "hf_datasets")),
            data_path=str(raw_data.get("data_path", "")),
            split=str(raw_data.get("split", "train")),
            image_root=str(raw_data.get("image_root", "")),
            max_samples=int(raw_data.get("max_samples", 0)),
        ),
    )

    # ── Validate ──
    _validate(cfg)

    # ── Merge method defaults ──
    _merge_method_defaults(cfg)

    return cfg


def _validate(cfg: TrainingConfig) -> None:
    """Validate config fields; raise ValueError with all issues at once."""
    errors: List[str] = []

    if not cfg.model.model_path:
        errors.append("model.model_path: required field is empty")

    if not cfg.data.data_path:
        errors.append("data.data_path: required field is empty")

    if cfg.runtime.mode not in ("local", "colab", "ssh"):
        errors.append(
            f"runtime.mode: '{cfg.runtime.mode}' is not valid. "
            f"Must be one of: local, colab, ssh"
        )

    if cfg.runtime.mode == "ssh":
        if not cfg.runtime.ssh.host:
            errors.append("runtime.ssh.host: required when mode is 'ssh'")
        if not cfg.runtime.ssh.username:
            errors.append("runtime.ssh.username: required when mode is 'ssh'")

    # Validate ft_method against registry (lazy import to avoid circular deps)
    try:
        from mmit.registry import registry
        available = registry.list("training_method")
        if available and cfg.training.ft_method not in available:
            errors.append(
                f"training.ft_method: '{cfg.training.ft_method}' is not registered. "
                f"Available: {available}"
            )
    except Exception:
        pass  # registry not yet populated (e.g. unit test without full init)

    if errors:
        msg = "Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def _merge_method_defaults(cfg: TrainingConfig) -> None:
    """Merge method default params into cfg.training.params."""
    try:
        from mmit.registry import registry
        method_obj = registry.build("training_method", cfg.training.ft_method)
        defaults = method_obj.default_config()
        # Warn about unknown user-provided keys
        unknown = set(cfg.training.params) - set(defaults)
        if unknown:
            warnings.warn(
                f"Unknown params for method '{cfg.training.ft_method}': {unknown}. "
                f"Known params: {set(defaults)}",
                stacklevel=3,
            )
        # Merge: user params override defaults
        merged = {**defaults, **cfg.training.params}
        cfg.training.params = merged
    except Exception:
        pass  # graceful fallback if registry not available


# ── Conversion ───────────────────────────────────────────────────────

def config_to_trainer_dict(cfg: TrainingConfig) -> dict:
    """Convert TrainingConfig to the flat dict format expected by __main__.py.

    Returns
    -------
    dict
        ``{"method": {...}, "training": {...}, "data": {...}}``
    """
    return {
        "method": {
            "model_path": cfg.model.model_path,
            "family": cfg.model.family,
        },
        "training": {
            "ft_method": cfg.training.ft_method,
            "num_epochs": cfg.training.num_epochs,
            "per_device_batch_size": cfg.training.per_device_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "learning_rate": cfg.training.learning_rate,
            "warmup_ratio": cfg.training.warmup_ratio,
            "weight_decay": cfg.training.weight_decay,
            "max_grad_norm": cfg.training.max_grad_norm,
            "save_steps": cfg.training.save_steps,
            "output_dir": cfg.training.output_dir,
            "params": cfg.training.params,
        },
        "data": {
            "adapter": cfg.data.adapter,
            "data_path": cfg.data.data_path,
            "split": cfg.data.split,
            "image_root": cfg.data.image_root,
            "max_samples": cfg.data.max_samples,
        },
    }
