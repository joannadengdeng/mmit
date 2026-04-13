"""SEEDBenchmark — SEED-Bench multi-dimensional visual understanding evaluation.

SEED-Bench evaluates 12 dimensions of visual understanding:
  Scene Understanding, Instance Identity, Instance Attributes,
  Instance Location, Instance Counting, Spatial Relation,
  Instance Interaction, Visual Reasoning, Text Understanding,
  Celebrity Recognition, Landmark Recognition, Chart Understanding

All questions are multiple-choice (A/B/C/D).

Input format (JSONL)::

    {"question_id": 1, "image": "SEED/img001.jpg",
     "text": "What is the man doing?",
     "choices": ["Running", "Swimming", "Reading", "Sleeping"],
     "answer": "A",
     "category": "Instance Interaction"}

Reference: Li et al., "SEED-Bench: Benchmarking Multimodal LLMs with
Generative Comprehension", CVPR 2024.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import normalize_answer


_INSTRUCTION = "Answer with the option's letter from the given choices directly."


class SEEDBenchmark(Benchmark):
    """SEED-Bench multi-dimensional MCQ benchmark.

    Parameters
    ----------
    question_file:
        Path to the JSONL question file.
    image_root:
        Root directory for images.
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
            # Format choices
            choices = q.get("choices", [])
            if isinstance(choices, list) and choices:
                options_text = "\n".join(
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                )
                full_question = q.get("text", q.get("question", "")) + "\n" + options_text
            else:
                full_question = q.get("text", q.get("question", ""))

            yield EvalSample(
                id=str(q.get("question_id", q.get("id", ""))),
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root and "image" in q else q.get("image", ""),
                question=full_question,
                ground_truth=q.get("answer", ""),
                metadata={"category": q.get("category", "")},
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        gt_map = {
            str(q.get("question_id", q.get("id", ""))): q
            for q in self._questions
        }

        # Overall accuracy
        correct = 0
        category_results: Dict[str, List[bool]] = defaultdict(list)

        for pred in predictions:
            qid = str(pred["id"])
            q = gt_map.get(qid, {})
            gt = str(q.get("answer", "")).strip().upper()

            pred_ans = normalize_answer(pred["prediction"]).strip().upper()
            if pred_ans and pred_ans[0] in "ABCDEFGH":
                pred_ans = pred_ans[0]

            is_correct = pred_ans == gt
            if is_correct:
                correct += 1

            category = q.get("category", "unknown")
            category_results[category].append(is_correct)

        accuracy = correct / len(predictions) if predictions else 0.0
        result = {"accuracy": round(accuracy * 100, 2)}

        # Per-category accuracy
        for cat, results in sorted(category_results.items()):
            cat_acc = sum(results) / len(results) if results else 0.0
            result[f"cat_{cat}"] = round(cat_acc * 100, 2)

        return result

    def __len__(self) -> int:
        return len(self._questions)
