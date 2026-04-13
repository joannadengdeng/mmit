"""ScienceQABenchmark — ScienceQA image subset multiple-choice evaluation.

Input format (HuggingFace dataset: lmms-lab/ScienceQA, config "ScienceQA-IMG")::

    {"question": "Which material is this ring made of?",
     "choices": ["copper", "iteite", "ite old", "iron"],
     "answer": 0,   # index into choices → "A"
     "hint": "The ring is shiny and reddish-brown.",
     "image": <PIL.Image>}

Reference: Lu et al., "Learn to Explain: Multimodal Reasoning via Thought Chains
for Science Question Answering", NeurIPS 2022.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import normalize_answer


_INSTRUCTION = "Answer with the option's letter from the given choices directly."


class ScienceQABenchmark(Benchmark):
    """ScienceQA image-subset multiple-choice benchmark.

    Parameters
    ----------
    question_file:
        Path to a JSONL file with questions, or None to load from HuggingFace.
    image_root:
        Root directory for images (if using file paths instead of PIL).
    split:
        Dataset split to evaluate on (default "test").
    max_samples:
        Limit number of questions for quick testing.
    """

    def __init__(
        self,
        question_file: Optional[str] = None,
        image_root: str = "",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        self.question_file = question_file
        self.image_root = image_root
        self.split = split
        self.max_samples = max_samples
        self._questions: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self.question_file and os.path.isfile(self.question_file):
            questions = []
            with open(self.question_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        questions.append(json.loads(line))
            return questions

        # Load from HuggingFace
        try:
            import datasets
            ds = datasets.load_dataset(
                "lmms-lab/ScienceQA", "ScienceQA-IMG",
                split=self.split, streaming=True,
            )
            questions = []
            for idx, row in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break
                # Convert answer index to letter
                answer_idx = row.get("answer", 0)
                answer_letter = chr(65 + int(answer_idx)) if isinstance(answer_idx, (int, float)) else str(answer_idx)

                q = {
                    "id": str(idx),
                    "question": row.get("question", ""),
                    "choices": row.get("choices", []),
                    "answer": answer_letter,
                    "hint": row.get("hint", ""),
                    "image": row.get("image"),
                }
                questions.append(q)
            return questions
        except Exception as e:
            raise RuntimeError(
                f"Cannot load ScienceQA: {e}. "
                f"Provide a question_file or install 'datasets'."
            )

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for q in self._questions:
            # Format choices into question text
            choices = q.get("choices", [])
            options_text = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(choices) if c
            )
            hint = q.get("hint", "").strip()
            parts = []
            if hint:
                parts.append(f"Context: {hint}")
            parts.append(q["question"])
            if options_text:
                parts.append(options_text)
            full_question = "\n".join(parts)

            # Handle image
            image_path = ""
            metadata = {}
            img = q.get("image")
            if img is not None:
                try:
                    from PIL import Image
                    if isinstance(img, Image.Image):
                        metadata["_pil_image"] = img
                        image_path = "<in_memory>"
                except ImportError:
                    pass

            yield EvalSample(
                id=str(q.get("id", "")),
                image_path=image_path,
                question=full_question,
                ground_truth=q.get("answer", ""),
                metadata=metadata,
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        gt_map = {str(q.get("id", "")): q.get("answer", "").strip().upper()
                  for q in self._questions}
        correct = 0
        for pred in predictions:
            pred_ans = normalize_answer(pred["prediction"]).strip().upper()
            if pred_ans and pred_ans[0] in "ABCDEFGH":
                pred_ans = pred_ans[0]
            gt = gt_map.get(str(pred["id"]), "")
            if pred_ans == gt:
                correct += 1
        accuracy = correct / len(predictions) if predictions else 0.0
        return {"accuracy": round(accuracy * 100, 2)}

    def __len__(self) -> int:
        return len(self._questions)
