"""MMBenchBenchmark — MMBench multiple-choice evaluation.

Input format (TSV with header row)::

    index  question  A  B  C  D  answer  image
    0      ...       ...  ...  ...  ...  A  <base64 or path>

The ``image`` column may be:
- A file path relative to ``image_root``
- A base64-encoded JPEG/PNG string (LLaVA eval uses this format)

Reference: Liu et al., "MMBench: Is Your Multi-modal Model an All-around Player?",
ECCV 2024.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from typing import Dict, Iterator, List, Optional

from mmit.data.types import EvalSample
from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.metrics.vqa import normalize_answer


_INSTRUCTION = "Answer with the option's letter from the given choices directly."


class MMBenchBenchmark(Benchmark):
    """MMBench multiple-choice benchmark.

    Parameters
    ----------
    tsv_file:
        Path to the MMBench TSV file.
    image_root:
        If image paths are relative (not base64), prepend this root.
    split:
        One of ``"dev"`` or ``"test"`` (informational only).
    """

    def __init__(
        self,
        tsv_file: str,
        image_root: str = "",
        split: str = "dev",
    ) -> None:
        self.tsv_file = tsv_file
        self.image_root = image_root
        self.split = split
        self._rows: List[Dict] = self._load_tsv()
        self._tmp_dir: Optional[str] = None  # for base64 image temp files

    def _load_tsv(self) -> List[Dict]:
        rows = []
        with open(self.tsv_file) as f:
            import csv
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
        return rows

    def _resolve_image(self, row: Dict) -> str:
        """Return an image file path, decoding base64 if necessary."""
        img_val = row.get("image", "")
        if not img_val:
            return ""
        # Check if it looks like base64
        if not os.path.splitext(img_val)[1] and len(img_val) > 260:
            # Decode base64 to a temp file
            if self._tmp_dir is None:
                self._tmp_dir = tempfile.mkdtemp(prefix="mmit_mmbench_")
            img_data = base64.b64decode(img_val)
            idx = row.get("index", "img")
            tmp_path = os.path.join(self._tmp_dir, f"{idx}.jpg")
            if not os.path.exists(tmp_path):
                with open(tmp_path, "wb") as f:
                    f.write(img_data)
            return tmp_path
        # File path
        if self.image_root:
            return os.path.join(self.image_root, img_val)
        return img_val

    # ------------------------------------------------------------------
    def iter_questions(self) -> Iterator[EvalSample]:
        for row in self._rows:
            options_text = "\n".join(
                f"{letter}. {row.get(letter, '')}"
                for letter in ["A", "B", "C", "D"]
                if row.get(letter, "").strip()
            )
            question = row.get("question", "") + "\n" + options_text
            img_path = self._resolve_image(row)
            yield EvalSample(
                id=str(row.get("index", "")),
                image_path=img_path,
                question=question,
                ground_truth=row.get("answer", ""),
                metadata={"category": row.get("category", "")},
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.question + "\n" + _INSTRUCTION

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        gt_map = {str(r.get("index", "")): r.get("answer", "").strip().upper()
                  for r in self._rows}
        correct = 0
        for pred in predictions:
            pred_ans = normalize_answer(pred["prediction"]).strip().upper()
            # Extract first letter if model outputs full option text
            if pred_ans and pred_ans[0] in "ABCD":
                pred_ans = pred_ans[0]
            gt = gt_map.get(str(pred["id"]), "")
            if pred_ans == gt:
                correct += 1
        accuracy = correct / len(predictions) if predictions else 0.0
        return {"accuracy": round(accuracy * 100, 2)}

    def __len__(self) -> int:
        return len(self._rows)
