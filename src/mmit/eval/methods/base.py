"""Method ABC — the central interface for multimodal inference in mmit."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from mmit.data.types import CanonicalSample, EvalSample


class Method(ABC):
    """Full multimodal inference pipeline for one model variant.

    Lifecycle
    ---------
    1. ``Method.from_pretrained(path)``  — easiest entry point
    2. ``method.prepare_input(sample)``  — per-sample preprocessing
    3. ``method.generate(prepared)``     — autoregressive decoding
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        family: Optional[str] = None,
        **kwargs,
    ) -> "Method":
        """Auto-detect model family and load from a HuggingFace checkpoint."""
        from mmit.eval.methods.hf_method import HFMethod

        if family is None:
            try:
                from mmit.eval.methods.hf_method import detect_model_family
                family = detect_model_family(model_path)
            except ValueError:
                family = None

        return HFMethod(model_path, family=family, **kwargs)

    @abstractmethod
    def prepare_input(
        self,
        sample: CanonicalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Return a dict of tensors ready for ``generate``."""

    def prepare_eval_input(
        self,
        sample: EvalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Convenience wrapper: build a CanonicalSample from an EvalSample."""
        from mmit.data.types import Turn
        cs = CanonicalSample(
            id=sample.id,
            image_path=sample.image_path,
            turns=[Turn(role="human", content=sample.question)],
            metadata=sample.metadata,
        )
        return self.prepare_input(cs, image_root=image_root)

    @abstractmethod
    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response string from preprocessed inputs."""
