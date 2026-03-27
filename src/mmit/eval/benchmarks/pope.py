"""POPEBenchmark — Polling-based Object Probing Evaluation (yes/no hallucination).

Input format (JSONL)::

    {"question_id": 1, "image": "COCO_val2014_000000131089.jpg",
     "text": "Is there a car in the image?", "label": "yes"}

Reference: Li et al., "Evaluating Object Hallucination in Large Vision-Language Models",
EMNLP 2023.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import normalize_answer


_INSTRUCTION = "Please answer yes or no."


class POPEBenchmark(Benchmark):
    """POPE yes/no hallucination benchmark.

    Parameters
    ----------
    question_file:
        Path to the JSONL question + label file.
    image_root:
        Root directory for images.
    """

    def __init__(self, question_file: str, image_root: str = "") -> None:
        self.question_file = question_file
        self.image_root = image_root
        self._questions: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        items = []
        with open(self.question_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for q in self._questions:
            yield EvalSample(
                id=str(q["question_id"]),
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root else q["image"],
                question=q["text"],
                ground_truth=q.get("label", ""),
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        gt_map = {str(q["question_id"]): q.get("label", "yes")
                  for q in self._questions}

        tp = fp = tn = fn = 0
        for pred in predictions:
            pred_ans = normalize_answer(pred["prediction"])
            pred_yes = pred_ans.startswith("yes")
            gt_yes = normalize_answer(gt_map.get(str(pred["id"]), "yes")).startswith("yes")
            if pred_yes and gt_yes:
                tp += 1
            elif pred_yes and not gt_yes:
                fp += 1
            elif not pred_yes and not gt_yes:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        yes_rate = (tp + fp) / total if total else 0.0

        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "yes_rate": round(yes_rate * 100, 2),
        }

    def __len__(self) -> int:
        return len(self._questions)
