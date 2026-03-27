"""Preprocessor ABC — converts CanonicalSample + HF Processor into model-ready tensors.

A Preprocessor handles the model-family-specific tokenization, image processing,
and label masking required for VLM training. It also provides a collation method
to batch multiple preprocessed samples.

Built-in implementations:
  - ChatTemplatePreprocessor — uses processor.apply_chat_template(), works for
    all HF VLMs that support chat templates (LLaVA, Qwen2-VL, Gemma, etc.)
  - MultiImagePreprocessor — handles multi-image inputs (e.g. CaD-VI)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


IGNORE_INDEX = -100


class Preprocessor(ABC):
    """Base class for all preprocessors.

    A Preprocessor converts a CanonicalSample into a dict of tensors ready
    for model.forward(), and provides a collate method for batching.
    """

    @abstractmethod
    def tokenize(
        self,
        sample: Any,
        processor: Any,
        image_root: str = "",
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        """Convert a single CanonicalSample into model-ready tensors.

        Parameters
        ----------
        sample : CanonicalSample
            The input sample with turns, image_path, metadata.
        processor :
            HuggingFace processor (tokenizer + image processor).
        image_root : str
            Root directory for resolving relative image paths.
        max_length : int
            Maximum sequence length for tokenization.

        Returns
        -------
        dict
            Must contain at minimum: ``input_ids``, ``labels``, ``attention_mask``.
            May also contain: ``pixel_values``, ``image_sizes``, etc.
        """

    @abstractmethod
    def collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of tokenized samples into a padded batch.

        Parameters
        ----------
        samples :
            List of dicts returned by ``tokenize()``.

        Returns
        -------
        dict
            Batched tensors ready for ``model(**batch)``.
        """

    def supports_multi_image(self) -> bool:
        """Whether this preprocessor handles multi-image samples."""
        return False
