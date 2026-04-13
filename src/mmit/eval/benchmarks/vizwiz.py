"""VizWizBenchmark — VizWiz VQA evaluation (images from blind users).

VizWiz images are often blurry, poorly lit, or partially occluded — testing
model robustness to real-world image quality. Uses VQA soft accuracy scoring.

Input format (JSONL)::

    {"question_id": 0, "image": "VizWiz_val_00000000.jpg",
     "text": "What is this?",
     "answers": ["milk", "milk", "milk carton", "milk", "milk", ...]}

Reference: Gurari et al., "VizWiz Grand Challenge: Answering Visual Questions
from Blind People", CVPR 2018.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import aggregate_vqa_accuracy


_INSTRUCTION = "Answer the question using a single word or phrase."


class VizWizBenchmark(Benchmark):
    """VizWiz VQA benchmark (noisy real-world images from blind users).

    Parameters
    ----------
    question_file:
        Path to the JSONL question file.
    image_root:
        Root directory for images.
    annotation_file:
        Optional separate annotation file. If None, uses answers from question_file.
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
            data = ann if isinstance(ann, list) else ann.get("annotations", ann.get("data", []))
            result = {}
            for item in data:
                qid = str(item.get("question_id", item.get("id", "")))
                answers = item.get("answers", [])
                if answers and isinstance(answers[0], dict):
                    result[qid] = [a.get("answer", str(a)) for a in answers]
                else:
                    result[qid] = [str(a) for a in answers]
            return result
        # Fall back to answers in question file
        return {
            str(q.get("question_id", q.get("id", ""))): q.get("answers", [])
            for q in self._questions
        }

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for q in self._questions:
            qid = str(q.get("question_id", q.get("id", "")))
            yield EvalSample(
                id=qid,
                image_path=os.path.join(self.image_root, q["image"])
                           if self.image_root else q["image"],
                question=q.get("text", q.get("question", "")),
                ground_truth=self._gt_map.get(qid, []),
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
