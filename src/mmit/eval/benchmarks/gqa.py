"""GQABenchmark — GQA compositional visual reasoning evaluation.

Input format (JSONL, one JSON object per line)::

    {"question_id": "07333408", "image": "n161313.jpg",
     "text": "Is the horse brown?",
     "answer": "yes"}

This is a short-answer benchmark using exact match after normalization.

Reference: Hudson & Manning, "GQA: A New Dataset for Real-World Visual Reasoning
and Compositional Question Answering", CVPR 2019.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import normalize_answer


_INSTRUCTION = "Answer the question using a single word or phrase."


class GQABenchmark(Benchmark):
    """GQA compositional visual reasoning benchmark.

    Parameters
    ----------
    question_file:
        Path to the JSONL question file.
    image_root:
        Root directory for images (prepended to image paths).
    """

    def __init__(
        self,
        question_file: str,
        image_root: str = "",
    ) -> None:
        self.question_file = question_file
        self.image_root = image_root
        self._questions: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        questions = []
        with open(self.question_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for q in self._questions:
            yield EvalSample(
                id=str(q.get("question_id", q.get("id", ""))),
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root else q["image"],
                question=q["text"] if "text" in q else q.get("question", ""),
                ground_truth=q.get("answer", ""),
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        gt_map = {
            str(q.get("question_id", q.get("id", ""))): q.get("answer", "")
            for q in self._questions
        }
        correct = 0
        for pred in predictions:
            pred_ans = normalize_answer(pred["prediction"])
            gt_ans = normalize_answer(gt_map.get(str(pred["id"]), ""))
            if pred_ans == gt_ans:
                correct += 1
        accuracy = correct / len(predictions) if predictions else 0.0
        return {"accuracy": round(accuracy * 100, 2)}

    def __len__(self) -> int:
        return len(self._questions)
