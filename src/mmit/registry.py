"""Global plugin registry for mmit components.

Usage
-----
from mmit.registry import registry
registry.register("training_method", "my-method", MyMethod)
method = registry.build("training_method", "qlora")
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


_SLOTS = (
    "dataset",           # DatasetAdapter: raw files → CanonicalSample
    "method",            # eval Method: inference wrappers
    "training_method",   # TrainingMethod: PEFT / freeze / full-ft strategies
    "preprocessor",      # Preprocessor: CanonicalSample → tokenized tensors
    "mixer",             # DataMixer: multi-source dataset mixing
    "filter",            # DataFilter: pre-training data quality filtering
    "loss",              # LossFunction: decoupled loss computation
)


class Registry:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Type]] = {s: {} for s in _SLOTS}
        self._defaults: Dict[str, Dict[str, Dict]] = {s: {} for s in _SLOTS}

    # ------------------------------------------------------------------
    def register(
        self,
        slot: str,
        name: str,
        cls: Type,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        if slot not in self._store:
            raise ValueError(f"Unknown slot '{slot}'. Valid slots: {_SLOTS}")
        self._store[slot][name] = cls
        self._defaults[slot][name] = defaults or {}

    def build(self, slot: str, name: str, **kwargs: Any) -> Any:
        if slot not in self._store:
            raise ValueError(f"Unknown slot '{slot}'.")
        if name not in self._store[slot]:
            available = list(self._store[slot])
            raise KeyError(f"'{name}' not found in slot '{slot}'. Available: {available}")
        # Merge registered defaults with caller-supplied kwargs (caller wins)
        merged = {**self._defaults[slot].get(name, {}), **kwargs}
        return self._store[slot][name](**merged)

    def get_cls(self, slot: str, name: str) -> Type:
        if name not in self._store[slot]:
            available = list(self._store[slot])
            raise KeyError(f"'{name}' not found in slot '{slot}'. Available: {available}")
        return self._store[slot][name]

    def list(self, slot: str) -> list:
        return list(self._store.get(slot, {}).keys())

    def __repr__(self) -> str:
        lines = ["Registry:"]
        for slot, entries in self._store.items():
            lines.append(f"  {slot}: {list(entries.keys())}")
        return "\n".join(lines)


# Singleton
registry = Registry()
