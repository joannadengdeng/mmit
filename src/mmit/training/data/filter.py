"""DataFilter — pre-training data quality filtering.

Filters operate on CanonicalSample and return True (keep) or False (discard).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mmit.data.types import CanonicalSample


class DataFilter(ABC):
    """Base class for data quality filters."""

    @abstractmethod
    def filter(self, sample: CanonicalSample) -> bool:
        """Return True to keep the sample, False to discard."""

    def score(self, sample: CanonicalSample) -> float:
        """Return a quality score in [0, 1]. Default: 1.0 if filter passes."""
        return 1.0 if self.filter(sample) else 0.0


class CompositeFilter(DataFilter):
    """Chain multiple filters with AND/OR logic.

    Parameters
    ----------
    filters : list of DataFilter
        The filters to chain.
    logic : str
        "and" (all must pass) or "or" (any passes).
    """

    def __init__(self, filters: List[DataFilter], logic: str = "and", **kwargs):
        self.filters = filters
        self.logic = logic.lower()

    def filter(self, sample: CanonicalSample) -> bool:
        if self.logic == "and":
            return all(f.filter(sample) for f in self.filters)
        elif self.logic == "or":
            return any(f.filter(sample) for f in self.filters)
        raise ValueError(f"Unknown logic: {self.logic}")


class TextLengthFilter(DataFilter):
    """Filter samples by text content length."""

    def __init__(self, min_length: int = 1, max_length: int = 10000, **kwargs):
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, sample: CanonicalSample) -> bool:
        total_len = sum(len(t.content) for t in sample.turns)
        return self.min_length <= total_len <= self.max_length
