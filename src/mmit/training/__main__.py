"""Headless CLI trainer — supports both legacy single-stage and new multi-stage configs.

Usage::

    # Multi-stage YAML config:
    python -m mmit.training --config configs/llava15_two_stage.yaml

    # Legacy single-stage (backward compatible):
    python -m mmit.training --config configs/local_qlora.yaml

    # JSON config (from subprocess):
    python -m mmit.training --config-json '{"model": {...}, "stages": [...]}'

Output format (one JSON object per line)::

    {"type":"status","data":{"status":"loading"}}
    {"type":"metric","data":{"stage":"s1","step":1,"loss":2.34,...}}
    {"type":"status","data":{"status":"completed"}}
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from mmit.training.stage_runner import StageRunner, StageConfig, _emit


def _legacy_to_stages(config: dict) -> tuple[str, str, list[StageConfig]]:
    """Convert legacy single-stage config dict to StageConfig list.

    Returns (model_path, model_family, stages).
    """
    method_cfg = config.get("method", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    model_path = method_cfg.get("model_path", "")
    model_family = method_cfg.get("family", "")

    # Build data source from legacy format
    data_source = {
        "adapter": data_cfg.get("adapter", "hf_datasets"),
        "data_path": data_cfg.get("data_path", ""),
        "split": data_cfg.get("split", "train"),
        "image_root": data_cfg.get("image_root", ""),
    }
    max_samples = data_cfg.get("max_samples", 0)
    if max_samples:
        data_source["max_samples"] = max_samples

    stage = StageConfig(
        name="training",
        data_sources=[data_source],
        mixer="concat",
        preprocessor="chat_template",
        training_method=training_cfg.get("ft_method", "qlora"),
        method_params=training_cfg.get("params", {}),
        loss="ce",
        num_epochs=training_cfg.get("num_epochs", 3),
        per_device_batch_size=training_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        save_steps=training_cfg.get("save_steps", 500),
        output_dir=training_cfg.get("output_dir", "output"),
    )
    return model_path, model_family, [stage]


def _parse_stages_config(config: dict) -> tuple[str, str, list[StageConfig]]:
    """Parse new multi-stage config dict.

    Returns (model_path, model_family, stages).
    """
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("model_path", "")
    model_family = model_cfg.get("family", "")

    stages = []
    for stage_raw in config.get("stages", []):
        data_cfg = stage_raw.get("data", {})
        sources = data_cfg.get("sources", [])
        # If no sources but has legacy data fields, convert
        if not sources and data_cfg.get("data_path"):
            sources = [{
                "adapter": data_cfg.get("adapter", "hf_datasets"),
                "data_path": data_cfg.get("data_path", ""),
                "split": data_cfg.get("split", "train"),
                "image_root": data_cfg.get("image_root", ""),
            }]

        training = stage_raw.get("training", {})
        stage = StageConfig(
            name=stage_raw.get("name", f"stage_{len(stages)}"),
            data_sources=sources,
            mixer=data_cfg.get("mixer", "concat"),
            filter_config=data_cfg.get("filter"),
            preprocessor=stage_raw.get("preprocessor", "chat_template"),
            preprocessor_params=stage_raw.get("preprocessor_params", {}),
            training_method=stage_raw.get("training_method", "qlora"),
            method_params=stage_raw.get("method_params", {}),
            loss=stage_raw.get("loss", "ce"),
            loss_params=stage_raw.get("loss_params", {}),
            num_epochs=training.get("num_epochs", 1),
            per_device_batch_size=training.get("per_device_batch_size", 4),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            learning_rate=training.get("learning_rate", 2e-5),
            warmup_ratio=training.get("warmup_ratio", 0.03),
            weight_decay=training.get("weight_decay", 0.0),
            max_grad_norm=training.get("max_grad_norm", 1.0),
            save_steps=training.get("save_steps", 500),
            output_dir=training.get("output_dir", "output"),
            resume_from=stage_raw.get("resume_from", ""),
        )
        stages.append(stage)

    return model_path, model_family, stages


def main():
    parser = argparse.ArgumentParser(description="mmit headless trainer")
    parser.add_argument("--config-json", default=None,
                        help="Full training config as JSON string")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    elif args.config_json:
        config = json.loads(args.config_json)
    else:
        parser.error("Either --config or --config-json is required")

    try:
        # Detect config format: new (has "stages") or legacy (has "training")
        if "stages" in config:
            model_path, family, stages = _parse_stages_config(config)
        else:
            model_path, family, stages = _legacy_to_stages(config)

        if not model_path:
            _emit("error", {"message": "model_path is required"})
            sys.exit(1)

        runner = StageRunner(model_path, model_family=family)
        runner.run(stages)

    except Exception as e:
        _emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        _emit("status", {"status": "failed"})
        sys.exit(1)


if __name__ == "__main__":
    main()
