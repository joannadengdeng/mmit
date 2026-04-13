"""MMEBenchmark — MME Perception and Cognition evaluation.

MME tests 14 subtasks across two categories:
  - Perception (10): existence, count, position, color, poster, celebrity,
                     scene, landmark, artwork, OCR
  - Cognition (4): commonsense, numerical, text translation, code reasoning

Each question is a yes/no format. Each subtask has pairs of questions where
one expects "yes" and the other expects "no" for the same image.

Scoring: per-subtask accuracy (both questions in a pair must be correct to score).
Total score = sum of all subtask scores (max ~2800).

Input format (JSONL or directory of text files)::

    {"question_id": "existence_001", "image": "exist/001.jpg",
     "text": "Is there a dog in the image?", "answer": "yes",
     "category": "existence"}

Reference: Fu et al., "MME: A Comprehensive Evaluation Benchmark for
Multimodal Large Language Models", 2023. arXiv:2306.13394
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import normalize_answer


_INSTRUCTION = "Answer the question using a single word or phrase."


class MMEBenchmark(Benchmark):
    """MME perception + cognition benchmark.

    Parameters
    ----------
    question_file:
        Path to a JSONL file containing all MME questions.
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
            yield EvalSample(
                id=str(q.get("question_id", q.get("id", ""))),
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root else q["image"],
                question=q.get("text", q.get("question", "")),
                ground_truth=q.get("answer", ""),
                metadata={"category": q.get("category", "")},
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        """Compute MME scores per subtask and totals.

        MME uses a paired scoring system: for each image, there is a yes-question
        and a no-question. Both must be answered correctly to score for that pair.
        Subtask score = (number of correct pairs) * 100 / (total pairs).

        Returns perception score, cognition score, and total.
        """
        gt_map = {
            str(q.get("question_id", q.get("id", ""))): q
            for q in self._questions
        }

        # Group by category
        category_results: Dict[str, List[bool]] = defaultdict(list)
        for pred in predictions:
            qid = str(pred["id"])
            q = gt_map.get(qid, {})
            category = q.get("category", "unknown")
            pred_ans = normalize_answer(pred["prediction"])
            gt_ans = normalize_answer(q.get("answer", ""))
            # Simple yes/no match
            pred_yes = pred_ans.startswith("yes")
            gt_yes = gt_ans.startswith("yes")
            category_results[category].append(pred_yes == gt_yes)

        # Compute per-subtask accuracy
        perception_cats = {
            "existence", "count", "position", "color", "poster",
            "celebrity", "scene", "landmark", "artwork", "OCR",
        }
        cognition_cats = {
            "commonsense_reasoning", "numerical_calculation",
            "text_translation", "code_reasoning",
        }

        subtask_scores = {}
        perception_total = 0.0
        cognition_total = 0.0

        for cat, results in category_results.items():
            acc = sum(results) / len(results) if results else 0.0
            score = acc * 100
            subtask_scores[cat] = round(score, 2)

            cat_lower = cat.lower().replace(" ", "_")
            if cat_lower in perception_cats or cat in perception_cats:
                perception_total += score
            elif cat_lower in cognition_cats or cat in cognition_cats:
                cognition_total += score

        total_score = perception_total + cognition_total

        return {
            "perception": round(perception_total, 2),
            "cognition": round(cognition_total, 2),
            "total": round(total_score, 2),
            **{f"subtask_{k}": v for k, v in subtask_scores.items()},
        }

    def __len__(self) -> int:
        return len(self._questions)
