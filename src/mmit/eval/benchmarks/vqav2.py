"""VQAv2Benchmark — VQA v2 validation / test evaluation.

Question file format (JSONL, LLaVA-style)::

    {"question_id": 262144000, "image": "COCO_val2014_000000262144.jpg",
     "text": "What is the color of the ball?"}

Annotation file format (official VQA v2 JSON)::

    {"annotations": [
        {"question_id": 262144000, "answers": [
            {"answer": "red"}, {"answer": "red"}, ...
        ]}
    ]}
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import aggregate_vqa_accuracy


_INSTRUCTION = "Answer the question using a single word or phrase."


class VQAv2Benchmark(Benchmark):
    """VQA v2 benchmark.

    Parameters
    ----------
    question_file:
        JSONL file with questions (one JSON object per line).
    image_root:
        Root directory prepended to image filenames.
    annotation_file:
        Official VQA v2 annotation JSON.  Required for ``score()``.
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

    def _load_questions(self) -> List[Dict]:
        questions = []
        with open(self.question_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    def _load_annotations(self) -> Dict:
        if self.annotation_file and os.path.isfile(self.annotation_file):
            with open(self.annotation_file) as f:
                ann = json.load(f)
            return {
                str(a["question_id"]): [ans["answer"] for ans in a["answers"]]
                for a in ann.get("annotations", [])
            }
        return {}

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for q in self._questions:
            yield EvalSample(
                id=str(q["question_id"]),
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root else q["image"],
                question=q["text"],
                ground_truth=self._gt_map.get(str(q["question_id"]), []),
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        if not self._gt_map:
            raise RuntimeError(
                "annotation_file is required for VQAv2Benchmark.score(). "
                "Please provide the official VQA v2 annotation JSON."
            )
        results = []
        for pred in predictions:
            qid = str(pred["id"])
            gts = self._gt_map.get(qid, [])
            results.append({"prediction": pred["prediction"], "ground_truths": gts})
        acc = aggregate_vqa_accuracy(results)
        return {"accuracy": round(acc * 100, 2)}

    def __len__(self) -> int:
        return len(self._questions)
