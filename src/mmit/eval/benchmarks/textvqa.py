"""TextVQABenchmark — evaluate on the TextVQA val set.

Input format (JSONL, one JSON object per line)::

    {"question_id": 0, "image": "train_images/000.jpg",
     "text": "What is written on the sign?",
     "answer_type": "other", "question_type": "what",
     "answers": ["STOP", "STOP", "stop", ...]}

This matches the ``llava_textvqa_val_v051_ocr.jsonl`` file used in the
official LLaVA evaluation.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import aggregate_vqa_accuracy


_INSTRUCTION = "Answer the question using a single word or phrase."


class TextVQABenchmark(Benchmark):
    """TextVQA val-set benchmark.

    Parameters
    ----------
    question_file:
        Path to the JSONL question file.
    image_root:
        Root directory for images (prepended to image paths).
    annotation_file:
        Optional path to the official annotation JSON for ground-truth answers.
        If not provided the answers embedded in the question file are used.
    """

    def __init__(
        self,
        question_file: str,
        image_root: str = "",
        annotation_file: Optional[str] = None,
    ) -> None:
        self.question_file = question_file
        self.image_root = image_root
        self.annotation_file = annotation_file
        self._questions: List[Dict] = self._load_questions()
        self._gt_map: Dict = self._load_annotations()

    # ------------------------------------------------------------------
    def _load_questions(self) -> List[Dict]:
        questions = []
        with open(self.question_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    def _load_annotations(self) -> Dict:
        """Build a map from question_id → list[answer] for scoring."""
        if self.annotation_file and os.path.isfile(self.annotation_file):
            with open(self.annotation_file) as f:
                ann = json.load(f)
            # Official TextVQA format: {"data": [{"question_id": ..., "answers": [...]}]}
            data = ann.get("data", ann)
            return {str(item["question_id"]): item["answers"] for item in data}
        # Fall back to answers embedded in the question file
        return {
            str(q["question_id"]): q.get("answers", [])
            for q in self._questions
        }

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for q in self._questions:
            yield EvalSample(
                id=str(q["question_id"]),
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root else q["image"],
                question=q["text"],
                ground_truth=self._gt_map.get(str(q["question_id"]), []),
                metadata={"question_type": q.get("question_type", "")},
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        results = []
        for pred in predictions:
            qid = str(pred["id"])
            gts = self._gt_map.get(qid, [])
            results.append({"prediction": pred["prediction"], "ground_truths": gts})
        acc = aggregate_vqa_accuracy(results)
        return {"accuracy": round(acc * 100, 2)}

    def __len__(self) -> int:
        return len(self._questions)
