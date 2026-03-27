"""Unified scoring — dispatch to task-specific metrics.

Supports multiple scoring metrics and an ``"auto"`` mode that picks the
most appropriate metric based on task type and ground-truth shape.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mmit.eval.metrics.vqa import (
    METRIC_REGISTRY,
    normalize_answer,
    vqa_accuracy,
)

# Metric keys the user can choose from (UI display order)
AVAILABLE_METRICS = list(METRIC_REGISTRY.keys())  # exact_match, vqa_accuracy, ...

# Label map for UI
METRIC_LABELS: Dict[str, str] = {k: v[1] for k, v in METRIC_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Auto-select logic
# ---------------------------------------------------------------------------

def auto_select_metric(
    task_type: str,
    ground_truths: Any = None,
) -> Tuple[str, str]:
    """Pick the best metric for the given task, return ``(metric_key, reason)``.

    Parameters
    ----------
    task_type:
        ``"open_vqa"``, ``"mcq"``, ``"yes_no"``, ``"caption"``, etc.
    ground_truths:
        The ground-truth value for a representative sample (used to detect
        whether multi-annotator data is available).
    """
    if task_type == "mcq":
        return "exact_match", "MCQ: 选项字母精确匹配即可"
    if task_type == "yes_no":
        return "exact_match", "Yes/No: 精确匹配"

    # open_vqa — check if multi-annotator data is available
    if task_type == "open_vqa":
        if isinstance(ground_truths, list) and len(ground_truths) >= 3:
            return "vqa_accuracy", "VQA v2 soft accuracy（检测到多人标注 >= 3）"
        return "exact_match", "Exact Match（单答案，VQA soft accuracy 会导致满分只有 0.33）"

    # classification (e.g. ImageNet)
    if task_type == "classification":
        return "exact_match", "Classification: 精确匹配类别名"

    # caption or unknown
    return "token_f1", "Caption/未知任务: Token F1 适合长文本对比"


# ---------------------------------------------------------------------------
# Single-metric scoring (backwards compatible)
# ---------------------------------------------------------------------------

def score_prediction(
    prediction: str,
    ground_truth: Any,
    task_type: str,
    metric: str = "auto",
) -> float:
    """Score a single prediction against ground truth.

    Returns a value in [0.0, 1.0].

    Parameters
    ----------
    prediction:
        Model output string.
    ground_truth:
        Ground-truth value — may be ``str``, ``int``, or ``List[str]``
        depending on *task_type*.
    task_type:
        One of ``"open_vqa"``, ``"mcq"``, ``"yes_no"``, ``"caption"``.
    metric:
        Scoring metric key, or ``"auto"`` to pick automatically.
    """
    if ground_truth is None or ground_truth == "":
        return 0.0

    # MCQ has its own special scoring
    if task_type == "mcq":
        return _score_mcq(prediction, ground_truth)

    # Resolve metric
    if metric == "auto":
        metric, _ = auto_select_metric(task_type, ground_truth)

    gts: List[str] = (
        ground_truth if isinstance(ground_truth, list)
        else [str(ground_truth)]
    )

    if metric in METRIC_REGISTRY:
        fn = METRIC_REGISTRY[metric][0]
        return fn(prediction, gts)

    # Fallback
    return 0.0


# ---------------------------------------------------------------------------
# Multi-metric scoring
# ---------------------------------------------------------------------------

def score_prediction_multi(
    prediction: str,
    ground_truth: Any,
    task_type: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Score with multiple metrics, return ``{metric_key: score}``.

    Parameters
    ----------
    metrics:
        List of metric keys. If ``None`` or contains ``"auto"``, the
        auto-selected metric is used.
    """
    if ground_truth is None or ground_truth == "":
        return {}

    if task_type == "mcq":
        return {"exact_match": _score_mcq(prediction, ground_truth)}

    gts: List[str] = (
        ground_truth if isinstance(ground_truth, list)
        else [str(ground_truth)]
    )

    # Resolve which metrics to compute
    if not metrics or metrics == ["auto"]:
        auto_key, _ = auto_select_metric(task_type, ground_truth)
        metric_keys = [auto_key]
    else:
        metric_keys = [m for m in metrics if m in METRIC_REGISTRY]
        if not metric_keys:
            auto_key, _ = auto_select_metric(task_type, ground_truth)
            metric_keys = [auto_key]

    result = {}
    for mk in metric_keys:
        fn = METRIC_REGISTRY[mk][0]
        result[mk] = fn(prediction, gts)
    return result


def _score_mcq(prediction: str, ground_truth: Any) -> float:
    """Score a multiple-choice prediction by letter match."""
    pred_norm = normalize_answer(prediction).strip().upper()
    gt_norm = str(ground_truth).strip().upper()

    # Extract first letter if model outputs full option text
    if pred_norm and pred_norm[0] in "ABCDEFGH":
        pred_letter = pred_norm[0]
    else:
        pred_letter = pred_norm

    # Ground truth might be an index (0, 1, 2, 3) -> convert to letter
    if gt_norm.isdigit():
        idx = int(gt_norm)
        gt_letter = chr(65 + idx) if 0 <= idx <= 7 else gt_norm
    else:
        gt_letter = gt_norm[0] if gt_norm and gt_norm[0] in "ABCDEFGH" else gt_norm

    return 1.0 if pred_letter == gt_letter else 0.0
