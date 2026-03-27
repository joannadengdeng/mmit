"""DataMixer — combines multiple data sources into a single training stream.

Each DataSource wraps a DatasetAdapter with optional weighting and
per-source instruction suffix (for response format prompting).
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Optional

from mmit.data.types import CanonicalSample, Turn


@dataclass
class DataSource:
    """A single data source with mixing configuration."""
    adapter: Any                     # DatasetAdapter instance
    weight: float = 1.0              # sampling probability weight
    instruction_suffix: str = ""     # appended to the last human turn
    max_samples: int = 0             # 0 = all samples


def _apply_suffix(sample: CanonicalSample, suffix: str) -> CanonicalSample:
    """Append instruction_suffix to the last human turn's content."""
    if not suffix:
        return sample
    new_turns = list(sample.turns)
    for i in range(len(new_turns) - 1, -1, -1):
        if new_turns[i].role == "human":
            new_turns[i] = Turn(
                role="human",
                content=new_turns[i].content.rstrip() + " " + suffix,
            )
            break
    return CanonicalSample(
        id=sample.id,
        image_path=sample.image_path,
        turns=new_turns,
        metadata=sample.metadata,
        instruction=sample.instruction,
    )


class DataMixer(ABC):
    """Base class for dataset mixing strategies."""

    @abstractmethod
    def mix(self, sources: List[DataSource]) -> List[CanonicalSample]:
        """Combine multiple sources into a single sample list."""

    @abstractmethod
    def __len__(self) -> int:
        ...


class ConcatMixer(DataMixer):
    """Simple concatenation of all sources."""

    def __init__(self, **kwargs):
        self._length = 0

    def mix(self, sources: List[DataSource]) -> List[CanonicalSample]:
        result = []
        for src in sources:
            samples = list(src.adapter)
            if src.max_samples > 0:
                samples = samples[:src.max_samples]
            for s in samples:
                result.append(_apply_suffix(s, src.instruction_suffix))
        self._length = len(result)
        return result

    def __len__(self) -> int:
        return self._length


class WeightedInterleaveMixer(DataMixer):
    """Weighted interleaved sampling from multiple sources.

    Each source is sampled with probability proportional to its weight.
    Used for LLaVA-1.5's 10-dataset mixing with per-dataset instruction suffixes.
    """

    def __init__(self, seed: int = 42, **kwargs):
        self._seed = seed
        self._length = 0

    def mix(self, sources: List[DataSource]) -> List[CanonicalSample]:
        # Materialize all sources
        pools = []
        for src in sources:
            samples = list(src.adapter)
            if src.max_samples > 0:
                samples = samples[:src.max_samples]
            samples = [_apply_suffix(s, src.instruction_suffix) for s in samples]
            pools.append(samples)

        if not pools:
            return []

        # Weighted interleave
        weights = [src.weight for src in sources]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        result = []
        indices = [0] * len(pools)
        rng = random.Random(self._seed)

        total_samples = sum(len(p) for p in pools)
        for _ in range(total_samples):
            # Pick a source based on weights
            src_idx = rng.choices(range(len(pools)), weights=probs, k=1)[0]
            if indices[src_idx] < len(pools[src_idx]):
                result.append(pools[src_idx][indices[src_idx]])
                indices[src_idx] += 1
            else:
                # This source is exhausted, try others
                for j in range(len(pools)):
                    if indices[j] < len(pools[j]):
                        result.append(pools[j][indices[j]])
                        indices[j] += 1
                        break

            # Check if all exhausted
            if all(indices[j] >= len(pools[j]) for j in range(len(pools))):
                break

        self._length = len(result)
        return result

    def __len__(self) -> int:
        return self._length
