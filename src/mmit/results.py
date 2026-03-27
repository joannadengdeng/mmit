"""ResultsManager — save, load, resume, and compute metrics for experiment runs."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """A single inference result."""
    id: str
    question: str
    prediction: str
    ground_truth: Any = None
    status: str = "ok"            # "ok" | "error"
    error: Optional[str] = None
    error_code: Optional[int] = None   # HTTP status code (e.g. 503)
    latency_s: float = 0.0
    usage: Optional[Dict[str, int]] = None  # {prompt_tokens, completion_tokens}
    request: Optional[Dict[str, Any]] = None  # raw API request payload
    score: Optional[float] = None  # 0.0~1.0, evaluation mode only
    scores: Optional[Dict[str, float]] = None  # multi-metric scores
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["usage"] is None:
            del d["usage"]
        if d["error"] is None:
            del d["error"]
        if d["error_code"] is None:
            del d["error_code"]
        if d["request"] is None:
            del d["request"]
        if d["score"] is None:
            del d["score"]
        if d["scores"] is None:
            del d["scores"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunHeader:
    """Metadata for an experiment run."""
    run_id: str = ""
    status: str = "running"       # "running" | "completed" | "failed"
    created_at: str = ""
    updated_at: str = ""
    model: Dict[str, str] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_total: Optional[int] = None
    completed_count: int = 0
    failed_count: int = 0
    metrics_summary: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Default results directory (mmit/results/ at repo root)
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "results"
)


# ---------------------------------------------------------------------------
# ResultsManager
# ---------------------------------------------------------------------------

class ResultsManager:
    """Manage a single experiment results file.

    Usage
    -----
    >>> mgr = ResultsManager.create(
    ...     model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    ...     model_family="qwen2_5_vl",
    ...     dataset_name="merve/vqav2-small",
    ...     split="validation",
    ...     parameters={"max_new_tokens": 256, "temperature": 0.2},
    ... )
    >>> mgr.add_prediction(PredictionRecord(id="0", question="...", prediction="..."))
    >>> mgr.save()
    >>> done = mgr.completed_ids()  # for resume
    """

    def __init__(
        self, path: str, header: RunHeader, predictions: List[PredictionRecord],
    ) -> None:
        self.path = path
        self.header = header
        self.predictions = predictions

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        model_id: str,
        model_family: str,
        dataset_name: str,
        split: str = "validation",
        max_samples: Optional[int] = None,
        streaming: bool = False,
        parameters: Optional[Dict[str, Any]] = None,
        expected_total: Optional[int] = None,
        results_dir: str = _DEFAULT_RESULTS_DIR,
        run_id: Optional[str] = None,
    ) -> "ResultsManager":
        """Create a new results file for an experiment run."""
        now = datetime.now()
        if run_id is None:
            ts = now.strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_id.rstrip("/"))
            ds_short = os.path.basename(dataset_name.rstrip("/"))
            run_id = f"{model_short}_{ds_short}_{ts}"

        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{run_id}.json")

        header = RunHeader(
            run_id=run_id,
            status="running",
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            model={"model_id": model_id, "family": model_family},
            dataset={
                "name": dataset_name,
                "split": split,
                "max_samples": max_samples,
                "streaming": streaming,
            },
            parameters=parameters or {},
            expected_total=expected_total,
        )
        mgr = cls(path=path, header=header, predictions=[])
        mgr.save()
        return mgr

    @classmethod
    def load(cls, path: str) -> "ResultsManager":
        """Load an existing results file."""
        with open(path) as f:
            data = json.load(f)
        header_d = data.get("header", {})
        header = RunHeader(**{
            k: v for k, v in header_d.items()
            if k in RunHeader.__dataclass_fields__
        })
        predictions = [
            PredictionRecord.from_dict(p)
            for p in data.get("predictions", [])
        ]
        return cls(path=path, header=header, predictions=predictions)

    @classmethod
    def find_resumable(
        cls,
        model_id: str,
        dataset_name: str,
        results_dir: str = _DEFAULT_RESULTS_DIR,
    ) -> Optional[str]:
        """Find the most recent incomplete run for this model+dataset.

        Returns the file path, or None if no resumable run exists.
        """
        if not os.path.isdir(results_dir):
            return None
        model_short = os.path.basename(model_id.rstrip("/"))
        ds_short = os.path.basename(dataset_name.rstrip("/"))
        prefix = f"{model_short}_{ds_short}_"

        candidates = []
        for fname in os.listdir(results_dir):
            if fname.startswith(prefix) and fname.endswith(".json"):
                fpath = os.path.join(results_dir, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if data.get("header", {}).get("status") == "running":
                        candidates.append(fpath)
                except (json.JSONDecodeError, OSError):
                    continue
        if not candidates:
            return None
        return sorted(candidates)[-1]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_prediction(self, record: PredictionRecord) -> None:
        """Append a single prediction record."""
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat()
        self.predictions.append(record)
        if record.status == "ok":
            self.header.completed_count += 1
        else:
            self.header.failed_count += 1

    def completed_ids(self) -> Set[str]:
        """Return sample IDs that have been successfully completed."""
        return {p.id for p in self.predictions if p.status == "ok"}

    def remove_errors(self) -> int:
        """Remove all error predictions so they can be retried.

        Returns the number of records removed.
        """
        before = len(self.predictions)
        self.predictions = [p for p in self.predictions if p.status != "error"]
        removed = before - len(self.predictions)
        self.header.failed_count = max(0, self.header.failed_count - removed)
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write the full results file to disk (atomic via tmp+rename)."""
        self.header.updated_at = datetime.now().isoformat()
        self.header.metrics_summary = self._compute_metrics_dict()

        data = {
            "header": asdict(self.header),
            "predictions": [p.to_dict() for p in self.predictions],
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def mark_completed(self) -> None:
        self.header.status = "completed"
        self.save()

    def mark_failed(self, error: str = "") -> None:
        self.header.status = "failed"
        self.save()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics_dict(self) -> Dict[str, Any]:
        ok_preds = [p for p in self.predictions if p.status == "ok"]
        if not ok_preds:
            return {}

        latencies = [p.latency_s for p in ok_preds if p.latency_s > 0]
        total_count = len(self.predictions)
        ok_count = len(ok_preds)

        metrics: Dict[str, Any] = {
            "success_rate": round(ok_count / total_count, 4) if total_count else 0,
        }
        if latencies:
            sorted_lat = sorted(latencies)
            p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
            metrics.update({
                "latency_mean_s": round(mean(latencies), 3),
                "latency_median_s": round(median(latencies), 3),
                "latency_p95_s": round(sorted_lat[p95_idx], 3),
                "latency_min_s": round(min(latencies), 3),
                "latency_max_s": round(max(latencies), 3),
                "total_time_s": round(sum(latencies), 3),
            })

        # Accuracy (evaluation mode)
        scored_preds = [p for p in ok_preds if p.score is not None]
        if scored_preds:
            metrics["accuracy"] = round(
                mean(p.score for p in scored_preds) * 100, 2
            )
            metrics["correct_count"] = sum(1 for p in scored_preds if p.score >= 1.0)
            metrics["wrong_count"] = sum(1 for p in scored_preds if p.score < 1.0)

        # Multi-metric aggregation
        multi_preds = [p for p in ok_preds if p.scores]
        if multi_preds:
            all_keys: set = set()
            for p in multi_preds:
                all_keys.update(p.scores.keys())
            metric_avgs = {}
            for mk in sorted(all_keys):
                vals = [p.scores[mk] for p in multi_preds if mk in p.scores]
                if vals:
                    metric_avgs[mk] = round(mean(vals) * 100, 2)
            if metric_avgs:
                metrics["metric_details"] = metric_avgs

        # Token usage (only if API provides it)
        usage_preds = [p for p in ok_preds if p.usage]
        if usage_preds:
            prompt_total = sum(p.usage.get("prompt_tokens", 0) for p in usage_preds)
            completion_total = sum(p.usage.get("completion_tokens", 0) for p in usage_preds)
            metrics["tokens_prompt_total"] = prompt_total
            metrics["tokens_completion_total"] = completion_total
            if latencies and completion_total > 0:
                metrics["tokens_per_second"] = round(
                    completion_total / sum(latencies), 2
                )
        return metrics

    def compute_metrics(self) -> Dict[str, Any]:
        """Return computed metrics without side effects."""
        return self._compute_metrics_dict()

    def format_metrics_display(self) -> str:
        """Format metrics as human-readable text for display."""
        m = self.compute_metrics()
        if not m:
            return "No results yet."

        total = self.header.completed_count + self.header.failed_count
        expected = self.header.expected_total

        lines = []
        if expected:
            lines.append(
                f"Progress: {total}/{expected} "
                f"({self.header.completed_count} ok, "
                f"{self.header.failed_count} errors)"
            )
        else:
            lines.append(
                f"Processed: {total} samples "
                f"({self.header.completed_count} ok, "
                f"{self.header.failed_count} errors)"
            )
        lines.append(f"Success rate: {m.get('success_rate', 0):.1%}")

        if "metric_details" in m and m["metric_details"]:
            from mmit.eval.metrics.scoring import METRIC_LABELS
            lines.append("★ Metrics:")
            for mk, mv in m["metric_details"].items():
                label = METRIC_LABELS.get(mk, mk)
                lines.append(f"  - {label}: {mv:.1f}%")
        elif "accuracy" in m:
            lines.append(
                f"★ Accuracy: {m['accuracy']:.1f}%  "
                f"(Correct: {m.get('correct_count', 0)}, "
                f"Wrong: {m.get('wrong_count', 0)})"
            )

        if "latency_mean_s" in m:
            lines.append(
                f"Latency: mean={m['latency_mean_s']:.2f}s, "
                f"median={m['latency_median_s']:.2f}s, "
                f"p95={m['latency_p95_s']:.2f}s"
            )
            lines.append(
                f"Latency range: [{m['latency_min_s']:.2f}s, {m['latency_max_s']:.2f}s]"
            )
            lines.append(f"Total time: {m['total_time_s']:.1f}s")

        if "tokens_prompt_total" in m:
            lines.append(
                f"Tokens: {m['tokens_prompt_total']} prompt, "
                f"{m['tokens_completion_total']} completion"
            )
            if "tokens_per_second" in m:
                lines.append(f"Throughput: {m['tokens_per_second']:.1f} tokens/s")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Listing (class-level)
    # ------------------------------------------------------------------

    @staticmethod
    def list_results(results_dir: str = _DEFAULT_RESULTS_DIR) -> List[Dict[str, Any]]:
        """List all result files with summary info."""
        if not os.path.isdir(results_dir):
            return []
        results = []
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                h = data.get("header", {})
                results.append({
                    "file": fname,
                    "path": fpath,
                    "run_id": h.get("run_id", fname),
                    "model_id": h.get("model", {}).get("model_id", "?"),
                    "dataset": h.get("dataset", {}).get("name", "?"),
                    "status": h.get("status", "?"),
                    "completed": h.get("completed_count", 0),
                    "failed": h.get("failed_count", 0),
                    "metrics": h.get("metrics_summary") or {},
                    "created_at": h.get("created_at", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return results
