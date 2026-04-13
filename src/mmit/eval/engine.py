"""EvalEngine — run a benchmark end-to-end and compute metrics."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.methods.base import Method


class EvalEngine:
    """Run a :class:`Benchmark` against a :class:`Method` and collect results.

    Example
    -------
    >>> from mmit import Method
    >>> from mmit.eval.engine import EvalEngine
    >>> from mmit.eval.benchmarks.textvqa import TextVQABenchmark
    >>>
    >>> method = Method.from_pretrained("liuhaotian/llava-v1.5-7b", device="cuda")
    >>> bench = TextVQABenchmark(
    ...     question_file="llava_textvqa_val_v051_ocr.jsonl",
    ...     image_root="train_images",
    ... )
    >>> engine = EvalEngine()
    >>> metrics = engine.run(method, bench, output_file="results/textvqa.jsonl")
    >>> print(metrics)  # {"accuracy": 58.3}
    """

    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        experiment_tracker=None,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tracker = experiment_tracker  # Optional ExperimentTracker

    # ------------------------------------------------------------------
    def run(
        self,
        method: Method,
        benchmark: Benchmark,
        output_file: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """Run inference over all benchmark questions and compute metrics.

        Parameters
        ----------
        method:
            A loaded :class:`Method` instance.
        benchmark:
            A :class:`Benchmark` instance providing questions.
        output_file:
            If provided, write JSONL predictions to this path.
        show_progress:
            Display a tqdm progress bar.

        Returns
        -------
        Dict of ``{metric_name: value}`` returned by ``benchmark.score()``.
        """
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        predictions: List[Dict] = []
        fout = open(output_file, "w") if output_file else None

        try:
            questions = list(benchmark.iter_questions())
            it = tqdm(questions, desc="Evaluating", disable=not show_progress)

            for sample in it:
                # Build the benchmark-specific prompt
                prompted_question = benchmark.build_prompt(sample)
                from mmit.data.types import EvalSample
                prompted_sample = EvalSample(
                    id=sample.id,
                    image_path=sample.image_path,
                    question=prompted_question,
                    ground_truth=sample.ground_truth,
                    metadata=sample.metadata,
                )

                prepared = method.prepare_eval_input(prompted_sample, image_root="")
                prediction = method.generate(
                    prepared,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )

                result = {
                    "id": sample.id,
                    "prediction": prediction,
                    "ground_truth": sample.ground_truth,
                }
                predictions.append(result)

                if fout:
                    fout.write(json.dumps(result) + "\n")
        finally:
            if fout:
                fout.close()

        metrics = benchmark.score(predictions)

        # Persist to experiment tracker (if available)
        if self._tracker is not None:
            bench_name = type(benchmark).__name__.replace("Benchmark", "").lower()
            self._tracker.log_eval(
                benchmark=bench_name,
                scores=metrics,
                predictions=predictions,
            )

        return metrics
