"""ExperimentTracker — unified experiment recording, comparison, and persistence.

Manages a single directory that contains ALL experiment data:
  experiments/
  ├── index.json                              ← registry of all experiments
  ├── exp_001_qlora_5k_20260409_143022/
  │   ├── config.yaml                         ← copy of training config
  │   ├── train/
  │   │   ├── metrics.jsonl                   ← step-by-step loss, lr, eta
  │   │   └── summary.json                    ← final avg_loss, time, params
  │   ├── eval/
  │   │   ├── vqav2_predictions.jsonl         ← per-sample predictions
  │   │   ├── vqav2_scores.json               ← {"accuracy": 76.7}
  │   │   ├── pope_predictions.jsonl
  │   │   ├── pope_scores.json
  │   │   └── ...
  │   ├── checkpoint/                         ← saved model weights
  │   │   ├── adapter_model.bin
  │   │   └── ...
  │   └── summary.json                        ← everything in one place
  └── exp_002_freeze_5k_20260409_150000/
      └── ...

Usage::

    from mmit.experiment import ExperimentTracker

    # Create a new experiment
    tracker = ExperimentTracker.create(
        base_dir="experiments",
        method="qlora",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        dataset="LLaVA-150K",
        num_samples=5000,
        config={"lora_r": 8, "lr": 2e-5},
    )

    # Record training metrics (called from stage_runner)
    tracker.log_train_step(step=1, loss=2.34, lr=2e-5, eta=3600)
    tracker.log_train_step(step=2, loss=2.10, lr=2e-5, eta=3500)

    # Record training summary
    tracker.log_train_summary(avg_loss=0.87, total_steps=2340, train_time_s=1200,
                              trainable_params=2_097_152, total_params=7_000_000_000)

    # Record eval results
    tracker.log_eval("vqav2", scores={"accuracy": 76.7},
                     predictions=[{"id": "001", "prediction": "stop", ...}])

    # Save checkpoint path
    tracker.set_checkpoint_path("experiments/exp_001_.../checkpoint/")

    # Finalize
    tracker.finalize()

    # Compare experiments
    from mmit.experiment import compare_experiments
    df = compare_experiments("experiments")
    print(df)
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    """Metadata for a single experiment."""
    exp_id: str = ""
    status: str = "running"           # running | completed | failed
    created_at: str = ""
    completed_at: str = ""
    # What
    method: str = ""                  # qlora, lora, freeze, mores, ...
    model: str = ""                   # Qwen/Qwen2.5-VL-3B-Instruct
    dataset: str = ""                 # LLaVA-150K, etc.
    num_samples: int = 0
    # Config
    config: Dict[str, Any] = field(default_factory=dict)
    # Training results
    train_summary: Dict[str, Any] = field(default_factory=dict)
    # Eval results (benchmark_name → scores dict)
    eval_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Paths
    checkpoint_path: str = ""
    exp_dir: str = ""
    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Track a single experiment: training metrics, eval scores, checkpoints.

    One tracker per experiment. Creates a directory with all data.
    """

    def __init__(self, meta: ExperimentMeta, base_dir: str):
        self.meta = meta
        self.base_dir = base_dir
        self._train_metrics_file: Optional[Any] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        base_dir: str = "experiments",
        method: str = "",
        model: str = "",
        dataset: str = "",
        num_samples: int = 0,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        exp_id: Optional[str] = None,
    ) -> "ExperimentTracker":
        """Create a new experiment with a unique ID and directory."""
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        model_short = os.path.basename(model.rstrip("/")).replace("-", "_")
        ds_short = os.path.basename(dataset.rstrip("/")).replace("-", "_")

        if exp_id is None:
            # Auto-generate: exp_001_qlora_5k_20260409_143022
            # Find next number
            os.makedirs(base_dir, exist_ok=True)
            existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
            next_num = len(existing) + 1
            samples_label = f"{num_samples // 1000}k" if num_samples >= 1000 else str(num_samples)
            exp_id = f"exp_{next_num:03d}_{method}_{samples_label}_{ts}"

        exp_dir = os.path.join(base_dir, exp_id)
        os.makedirs(os.path.join(exp_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoint"), exist_ok=True)

        meta = ExperimentMeta(
            exp_id=exp_id,
            status="running",
            created_at=now.isoformat(),
            method=method,
            model=model,
            dataset=dataset,
            num_samples=num_samples,
            config=config or {},
            tags=tags or [],
            notes=notes,
            exp_dir=exp_dir,
        )

        tracker = cls(meta=meta, base_dir=base_dir)

        # Save config as YAML (if pyyaml available) or JSON
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config or {}, f, indent=2, ensure_ascii=False)

        # Update index
        tracker._update_index()
        return tracker

    @classmethod
    def load(cls, exp_dir: str) -> "ExperimentTracker":
        """Load an existing experiment from its directory."""
        summary_path = os.path.join(exp_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"No summary.json in {exp_dir}")
        with open(summary_path) as f:
            data = json.load(f)
        meta = ExperimentMeta(**{
            k: v for k, v in data.items()
            if k in ExperimentMeta.__dataclass_fields__
        })
        base_dir = str(Path(exp_dir).parent)
        return cls(meta=meta, base_dir=base_dir)

    # ------------------------------------------------------------------
    # Training logging
    # ------------------------------------------------------------------

    def log_train_step(
        self,
        step: int,
        loss: float,
        lr: float = 0.0,
        avg_loss: float = 0.0,
        epoch: int = 0,
        eta: float = 0.0,
        **extra_metrics,
    ) -> None:
        """Append one training step's metrics to train/metrics.jsonl."""
        metrics_path = os.path.join(self.meta.exp_dir, "train", "metrics.jsonl")
        record = {
            "step": step,
            "loss": round(loss, 6),
            "avg_loss": round(avg_loss, 6),
            "lr": lr,
            "epoch": epoch,
            "eta": round(eta, 1),
            "timestamp": datetime.now().isoformat(),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in extra_metrics.items()},
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_train_summary(
        self,
        avg_loss: float = 0.0,
        total_steps: int = 0,
        train_time_s: float = 0.0,
        trainable_params: int = 0,
        total_params: int = 0,
        **extra,
    ) -> None:
        """Save training summary to train/summary.json."""
        summary = {
            "avg_loss": round(avg_loss, 6),
            "total_steps": total_steps,
            "train_time_s": round(train_time_s, 1),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": round(100 * trainable_params / max(1, total_params), 4),
            **extra,
        }
        self.meta.train_summary = summary
        summary_path = os.path.join(self.meta.exp_dir, "train", "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self._save_summary()

    # ------------------------------------------------------------------
    # Eval logging
    # ------------------------------------------------------------------

    def log_eval(
        self,
        benchmark: str,
        scores: Dict[str, float],
        predictions: Optional[List[Dict]] = None,
    ) -> None:
        """Record evaluation results for one benchmark.

        Parameters
        ----------
        benchmark:
            Name like "vqav2", "pope", "textvqa", etc.
        scores:
            Dict of metric_name → value, e.g. {"accuracy": 76.7}
        predictions:
            Optional list of per-sample predictions (saved as JSONL).
        """
        eval_dir = os.path.join(self.meta.exp_dir, "eval")

        # Save scores
        scores_path = os.path.join(eval_dir, f"{benchmark}_scores.json")
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)

        # Save predictions
        if predictions:
            preds_path = os.path.join(eval_dir, f"{benchmark}_predictions.jsonl")
            with open(preds_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # Update meta
        self.meta.eval_results[benchmark] = scores
        self._save_summary()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def set_checkpoint_path(self, path: str) -> None:
        """Record where the trained checkpoint was saved."""
        self.meta.checkpoint_path = path
        self._save_summary()

    def get_checkpoint_dir(self) -> str:
        """Return the checkpoint subdirectory for this experiment."""
        return os.path.join(self.meta.exp_dir, "checkpoint")

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self, status: str = "completed") -> None:
        """Mark experiment as completed and save final summary."""
        self.meta.status = status
        self.meta.completed_at = datetime.now().isoformat()
        self._save_summary()
        self._update_index()

    def fail(self, error: str = "") -> None:
        """Mark experiment as failed."""
        self.meta.notes = error if error else self.meta.notes
        self.finalize(status="failed")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_summary(self) -> None:
        """Write summary.json with all metadata."""
        summary_path = os.path.join(self.meta.exp_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(asdict(self.meta), f, indent=2, ensure_ascii=False)

    def _update_index(self) -> None:
        """Update the global index.json with this experiment."""
        index_path = os.path.join(self.base_dir, "index.json")

        # Load existing index
        index: List[Dict] = []
        if os.path.isfile(index_path):
            with open(index_path) as f:
                try:
                    index = json.load(f)
                except json.JSONDecodeError:
                    index = []

        # Update or append this experiment
        found = False
        entry = self._make_index_entry()
        for i, e in enumerate(index):
            if e.get("exp_id") == self.meta.exp_id:
                index[i] = entry
                found = True
                break
        if not found:
            index.append(entry)

        # Write atomically
        tmp = index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        os.replace(tmp, index_path)

    def _make_index_entry(self) -> Dict:
        """Create a compact index entry for this experiment."""
        # Flatten eval results for easy comparison
        eval_flat = {}
        for bench, scores in self.meta.eval_results.items():
            for metric, val in scores.items():
                eval_flat[f"{bench}_{metric}"] = val

        return {
            "exp_id": self.meta.exp_id,
            "status": self.meta.status,
            "method": self.meta.method,
            "model": self.meta.model,
            "dataset": self.meta.dataset,
            "num_samples": self.meta.num_samples,
            "created_at": self.meta.created_at,
            "completed_at": self.meta.completed_at,
            "train_loss": self.meta.train_summary.get("avg_loss"),
            "train_time_s": self.meta.train_summary.get("train_time_s"),
            "trainable_params": self.meta.train_summary.get("trainable_params"),
            **eval_flat,
            "tags": self.meta.tags,
            "exp_dir": self.meta.exp_dir,
        }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_experiments(
    base_dir: str = "experiments",
    filter_method: Optional[str] = None,
    filter_status: str = "completed",
    sort_by: Optional[str] = None,
) -> List[Dict]:
    """Load all experiments and return a comparison table.

    Parameters
    ----------
    base_dir:
        Root experiments directory.
    filter_method:
        Only include experiments with this method (e.g. "qlora").
    filter_status:
        Only include experiments with this status (default "completed").
    sort_by:
        Column to sort by (e.g. "vqav2_accuracy"). Descending.

    Returns
    -------
    List of dicts, one per experiment, with flattened metrics.

    Example output::

        [
            {"exp_id": "exp_001_qlora_5k_...", "method": "qlora",
             "vqav2_accuracy": 76.7, "pope_accuracy": 82.8, ...},
            {"exp_id": "exp_002_freeze_5k_...", "method": "freeze",
             "vqav2_accuracy": 80.6, "pope_accuracy": 83.6, ...},
        ]
    """
    index_path = os.path.join(base_dir, "index.json")
    if not os.path.isfile(index_path):
        return []

    with open(index_path) as f:
        index = json.load(f)

    # Filter
    results = []
    for entry in index:
        if filter_status and entry.get("status") != filter_status:
            continue
        if filter_method and entry.get("method") != filter_method:
            continue
        results.append(entry)

    # Sort
    if sort_by and results:
        results.sort(key=lambda e: e.get(sort_by, 0) or 0, reverse=True)

    return results


def print_comparison(
    base_dir: str = "experiments",
    benchmarks: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Print a formatted comparison table of experiments.

    Returns the table as a string.
    """
    results = compare_experiments(base_dir, **kwargs)
    if not results:
        return "No experiments found."

    if benchmarks is None:
        # Auto-detect benchmarks from results
        all_keys = set()
        for r in results:
            all_keys.update(k for k in r if k.endswith("_accuracy"))
        benchmarks = sorted(all_keys)

    # Build table
    lines = []
    # Header
    header = f"{'exp_id':<40} {'method':<10} {'samples':>7} {'loss':>6}"
    for b in benchmarks:
        short = b.replace("_accuracy", "")
        header += f" {short:>8}"
    header += f" {'time':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for r in results:
        row = f"{r.get('exp_id', '?'):<40} {r.get('method', '?'):<10} {r.get('num_samples', 0):>7}"
        loss = r.get("train_loss")
        row += f" {loss:>6.3f}" if loss else f" {'—':>6}"
        for b in benchmarks:
            val = r.get(b)
            row += f" {val:>8.1f}" if val is not None else f" {'—':>8}"
        time_s = r.get("train_time_s")
        if time_s:
            mins = int(time_s) // 60
            row += f" {mins:>5}m"
        else:
            row += f" {'—':>6}"
        lines.append(row)

    return "\n".join(lines)


def find_best(
    base_dir: str = "experiments",
    metric: str = "vqav2_accuracy",
    **kwargs,
) -> Optional[Dict]:
    """Find the experiment with the highest value for a given metric."""
    results = compare_experiments(base_dir, **kwargs)
    if not results:
        return None
    return max(results, key=lambda e: e.get(metric, 0) or 0)
