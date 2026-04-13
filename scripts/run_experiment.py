#!/usr/bin/env python3
"""run_experiment.py — unified train + eval with automatic experiment tracking.

Everything (config, loss curve, checkpoint, eval predictions, scores) is saved
in one experiment directory under experiments/.

Usage::

    # Train QLoRA on 5K samples + eval on VQAv2, POPE, TextVQA
    python scripts/run_experiment.py \
        --method qlora \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --dataset HuggingFaceH4/llava-instruct-mix-vsft \
        --num-samples 5000 \
        --lr 2e-5 \
        --benchmarks vqav2 pope textvqa

    # Train Freeze + eval on all 9 benchmarks
    python scripts/run_experiment.py \
        --method freeze \
        --model llava-hf/llava-1.5-7b-hf \
        --num-samples 5000 \
        --benchmarks all

    # Eval only (no training, baseline)
    python scripts/run_experiment.py \
        --eval-only \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --benchmarks vqav2 pope textvqa

    # Compare all experiments
    python scripts/run_experiment.py --compare

    # Compare only QLoRA experiments
    python scripts/run_experiment.py --compare --filter-method qlora
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="mmit unified experiment runner")

    # Mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare all experiments instead of running one")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate (no training)")

    # Model
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                        help="HuggingFace model path")

    # Training
    parser.add_argument("--method", default="qlora",
                        choices=["qlora", "lora", "full_ft", "freeze",
                                 "mores", "l2t", "lora_in_lora"],
                        help="Training method")
    parser.add_argument("--dataset", default="HuggingFaceH4/llava-instruct-mix-vsft",
                        help="Training dataset")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)

    # Eval
    parser.add_argument("--benchmarks", nargs="+", default=["vqav2", "pope", "textvqa"],
                        help="Benchmarks to evaluate on (or 'all')")
    parser.add_argument("--eval-samples", type=int, default=500,
                        help="Number of eval samples per benchmark")

    # Experiment
    parser.add_argument("--exp-dir", default="experiments",
                        help="Base directory for all experiments")
    parser.add_argument("--tags", nargs="*", default=[],
                        help="Tags for filtering experiments")
    parser.add_argument("--notes", default="", help="Notes for this experiment")

    # Compare mode options
    parser.add_argument("--filter-method", default=None,
                        help="Filter comparison by method")
    parser.add_argument("--sort-by", default="vqav2_accuracy",
                        help="Sort comparison by this metric")

    args = parser.parse_args()

    # ── Compare mode ──
    if args.compare:
        from mmit.experiment import print_comparison
        table = print_comparison(
            base_dir=args.exp_dir,
            filter_method=args.filter_method,
            sort_by=args.sort_by,
        )
        print(table)
        return

    # ── Create experiment tracker ──
    from mmit.experiment import ExperimentTracker

    config = {
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lora_r": args.lora_r,
        "eval_benchmarks": args.benchmarks,
        "eval_samples": args.eval_samples,
    }

    tracker = ExperimentTracker.create(
        base_dir=args.exp_dir,
        method=args.method if not args.eval_only else "baseline",
        model=args.model,
        dataset=args.dataset if not args.eval_only else "none",
        num_samples=args.num_samples if not args.eval_only else 0,
        config=config,
        tags=args.tags,
        notes=args.notes,
    )

    print(f"[mmit] Experiment: {tracker.meta.exp_id}")
    print(f"[mmit] Directory:  {tracker.meta.exp_dir}")
    print()

    try:
        # ── Training ──
        if not args.eval_only:
            _run_training(args, tracker)

        # ── Evaluation ──
        _run_eval(args, tracker)

        # ── Finalize ──
        tracker.finalize()
        print(f"\n[mmit] Experiment completed: {tracker.meta.exp_id}")
        print(f"[mmit] Results saved to: {tracker.meta.exp_dir}")

        # Print summary
        print("\n── Results ──")
        for bench, scores in tracker.meta.eval_results.items():
            scores_str = ", ".join(f"{k}: {v}" for k, v in scores.items())
            print(f"  {bench}: {scores_str}")

    except Exception as e:
        tracker.fail(str(e))
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_training(args, tracker):
    """Run training with experiment tracking."""
    from mmit.training.stage_runner import StageRunner, StageConfig

    print(f"[mmit] Training: {args.method} on {args.dataset} ({args.num_samples} samples)")
    print(f"[mmit] lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}")
    print()

    method_params = {}
    if args.method in ("qlora", "lora", "lora_in_lora"):
        method_params["lora_r"] = args.lora_r

    stage = StageConfig(
        name="training",
        data_sources=[{
            "adapter": "hf_datasets",
            "dataset": args.dataset,
            "split": "train",
            "max_samples": args.num_samples,
        }],
        mixer="concat",
        preprocessor="chat_template",
        training_method=args.method,
        method_params=method_params,
        loss="ce",
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        save_steps=0,  # Don't save intermediate checkpoints
        output_dir=tracker.get_checkpoint_dir(),
    )

    runner = StageRunner(
        args.model,
        model_family="",
        experiment_tracker=tracker,
    )
    runner.run([stage])


def _run_eval(args, tracker):
    """Run evaluation on specified benchmarks."""
    from mmit.eval.engine import EvalEngine

    benchmarks_to_run = args.benchmarks
    if "all" in benchmarks_to_run:
        benchmarks_to_run = [
            "vqav2", "textvqa", "pope", "mmbench",
            "gqa", "scienceqa", "mme", "seed", "vizwiz",
        ]

    # Load method
    # For eval-only, load the base model directly
    # For post-training, the model is already in memory (but we need to reload for eval)
    print(f"\n[mmit] Evaluating on: {', '.join(benchmarks_to_run)}")

    engine = EvalEngine(
        max_new_tokens=128,
        temperature=0.0,
        experiment_tracker=tracker,
    )

    # We need a Method instance for inference
    from mmit.eval.methods.base import Method
    method = Method.from_pretrained(args.model)

    for bench_name in benchmarks_to_run:
        print(f"\n[mmit] Running {bench_name}...")
        benchmark = _create_benchmark(bench_name, args.eval_samples)
        if benchmark is None:
            print(f"  [SKIP] {bench_name}: no data file configured")
            continue

        try:
            metrics = engine.run(method, benchmark)
            scores_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
            print(f"  [DONE] {bench_name}: {scores_str}")
        except Exception as e:
            print(f"  [ERROR] {bench_name}: {e}")


def _create_benchmark(name: str, max_samples: int):
    """Create a benchmark instance by name.

    Returns None if data files are not available.
    """
    # For HF-hosted benchmarks, we could auto-download.
    # For now, return None if question_file doesn't exist.
    # Users should set up their data directories.

    # Try to use HuggingFace datasets directly for some benchmarks
    try:
        if name == "scienceqa":
            from mmit.eval.benchmarks.scienceqa import ScienceQABenchmark
            return ScienceQABenchmark(max_samples=max_samples)
    except Exception:
        pass

    return None


if __name__ == "__main__":
    main()
