"""Benchmark ABC — unified interface for all eval benchmarks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List

from mmit.data.types import EvalSample


class Benchmark(ABC):
    """Base class for all evaluation benchmarks.

    Subclasses must implement:
    - ``iter_questions()``   — yield :class:`EvalSample` objects
    - ``build_prompt()``     — add benchmark-specific instruction suffixes
    - ``score()``            — compute metrics from predictions
    """

    @abstractmethod
    def iter_questions(self) -> Iterator[EvalSample]:
        """Yield evaluation questions one by one."""

    @abstractmethod
    def build_prompt(self, sample: EvalSample) -> str:
        """Return the question text with any benchmark-specific instruction suffix.

        Example suffixes:
          VQAv2:    "Answer the question using a single word or phrase."
          MMBench:  "Answer with the option's letter from the given choices directly."
          POPE:     "Please answer yes or no."
        """

    @abstractmethod
    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics.

        Parameters
        ----------
        predictions:
            List of dicts with at least ``{"id": ..., "prediction": str}``.

        Returns
        -------
        Dict mapping metric name → float, e.g. ``{"accuracy": 0.583}``.
        """

    def __len__(self) -> int:
        """Number of evaluation questions (override for efficiency)."""
        return sum(1 for _ in self.iter_questions())
