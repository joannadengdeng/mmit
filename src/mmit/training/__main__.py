"""Headless CLI trainer — multi-stage config only.

Usage::

    # YAML config:
    python -m mmit.training --config configs/llava15_two_stage.yaml

    # JSON config (from subprocess):
    python -m mmit.training --config-json '{"model": {...}, "stages": [...]}'

Config schema::

    model:
      model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
      family: "qwen2_5_vl"     # optional, auto-detected
    stages:
      - name: "stage1"
        data:
          sources:
            - adapter: "hf_datasets"
              dataset: "..."
              split: "train"
        training_method: "qlora"
        method_params: {lora_r: 8}
        loss: "ce"
        training:
          num_epochs: 1
          learning_rate: 2e-5
          per_device_batch_size: 4

Single-stage training: just put one element in `stages`.

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


def _parse_stages_config(config: dict) -> tuple[str, str, list[StageConfig]]:
    """Parse the multi-stage config dict into a list of StageConfig.

    Returns (model_path, model_family, stages).
    """
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("model_path", "")
    model_family = model_cfg.get("family", "")

    stages = []
    for stage_raw in config.get("stages", []):
        data_cfg = stage_raw.get("data", {})
        sources = data_cfg.get("sources", [])
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
        if "stages" not in config:
            _emit("error", {"message": "config must contain 'stages' key"})
            sys.exit(1)

        model_path, family, stages = _parse_stages_config(config)

        if not model_path:
            _emit("error", {"message": "model.model_path is required"})
            sys.exit(1)

        runner = StageRunner(model_path, model_family=family)
        runner.run(stages)

    except Exception as e:
        _emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        _emit("status", {"status": "failed"})
        sys.exit(1)


if __name__ == "__main__":
    main()
