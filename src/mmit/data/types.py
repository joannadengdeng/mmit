"""Core data types shared across all mmit components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    role: str     # "human" | "assistant"
    content: str


@dataclass
class CanonicalSample:
    """Unified training/inference sample used throughout mmit.

    image_path is relative to an ``image_root`` supplied at load time.
    ``instruction`` holds the fully-formatted prompt for inference (question +
    choices + instruction suffix).  When empty, callers should fall back to
    :pyattr:`first_question`.
    """
    id: str
    image_path: str
    turns: List[Turn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    instruction: str = ""

    @property
    def first_question(self) -> str:
        for t in self.turns:
            if t.role == "human":
                return t.content
        return ""

    @property
    def first_answer(self) -> str:
        for t in self.turns:
            if t.role == "assistant":
                return t.content
        return ""


@dataclass
class EvalSample:
    """A single benchmark question, optionally with ground-truth for scoring."""
    id: str
    image_path: str
    question: str
    ground_truth: Optional[Any] = None   # list of answers (VQA) or letter (MCQ)
    metadata: Dict[str, Any] = field(default_factory=dict)
