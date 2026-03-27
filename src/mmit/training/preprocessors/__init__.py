"""Preprocessors: CanonicalSample + HF Processor → model-ready tensors."""

from mmit.training.preprocessors.base import Preprocessor
from mmit.training.preprocessors.chat_template import ChatTemplatePreprocessor
from mmit.training.preprocessors.multi_image import MultiImagePreprocessor

__all__ = ["Preprocessor", "ChatTemplatePreprocessor", "MultiImagePreprocessor"]
