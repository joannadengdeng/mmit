"""mmit — Multimodal Instruction Tuning library.

Quick start
-----------
>>> from mmit import Method
>>> method = Method.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
>>> result = method.inference("What is this?", "image.jpg")
"""
from __future__ import annotations

from mmit.registry import registry

# ---------- Dataset adapters ----------
from mmit.data.adapters.llava_json import LLaVAJSONAdapter
from mmit.data.adapters.hf_datasets import HFDatasetsAdapter, DatasetProfile
registry.register("dataset", "llava_json", LLaVAJSONAdapter)
registry.register("dataset", "hf_datasets", HFDatasetsAdapter)

# ---------- Eval methods ----------
from mmit.eval.methods.hf_method import HFMethod
from mmit.eval.methods.litellm_method import LiteLLMMethod
registry.register("method", "hf_method", HFMethod)
registry.register("method", "litellm", LiteLLMMethod)

# ---------- Training Methods ----------
from mmit.training.methods import (
    QLoRAMethod, LoRAMethod, FullFTMethod,
    FreezeTuningMethod, L2TMethod, MoReSMethod,
)
from mmit.training.methods.lora_in_lora import LoRAInLoRAMethod
registry.register("training_method", "qlora", QLoRAMethod)
registry.register("training_method", "lora", LoRAMethod)
registry.register("training_method", "full_ft", FullFTMethod)
registry.register("training_method", "freeze", FreezeTuningMethod)
registry.register("training_method", "l2t", L2TMethod)
registry.register("training_method", "mores", MoReSMethod)
registry.register("training_method", "lora_in_lora", LoRAInLoRAMethod)

# ---------- Preprocessors ----------
from mmit.training.preprocessors import ChatTemplatePreprocessor, MultiImagePreprocessor
registry.register("preprocessor", "chat_template", ChatTemplatePreprocessor)
registry.register("preprocessor", "multi_image", MultiImagePreprocessor)

# ---------- Data Mixers ----------
from mmit.training.data.mixer import ConcatMixer, WeightedInterleaveMixer
registry.register("mixer", "concat", ConcatMixer)
registry.register("mixer", "weighted_interleave", WeightedInterleaveMixer)

# ---------- Data Filters ----------
from mmit.training.data.filter import CompositeFilter, TextLengthFilter
registry.register("filter", "composite", CompositeFilter)
registry.register("filter", "text_length", TextLengthFilter)

# ---------- Loss Functions ----------
from mmit.training.losses import CrossEntropyLoss, CEPlusOrthoLoss
registry.register("loss", "ce", CrossEntropyLoss)
registry.register("loss", "ce_ortho", CEPlusOrthoLoss)

# ---------- Results ----------
from mmit.results import PredictionRecord, ResultsManager

# ---------- Public API ----------
from mmit.eval.methods.base import Method
from mmit.data.types import CanonicalSample, EvalSample, Turn

__all__ = [
    "registry",
    "Method", "HFMethod", "LiteLLMMethod",
    "ResultsManager", "PredictionRecord",
    "CanonicalSample", "EvalSample", "Turn",
    "HFDatasetsAdapter", "DatasetProfile", "LLaVAJSONAdapter",
    "QLoRAMethod", "LoRAMethod", "DoRAMethod", "FullFTMethod",
    "FreezeTuningMethod", "L2TMethod", "MoReSMethod", "LoRAInLoRAMethod",
    "ChatTemplatePreprocessor", "MultiImagePreprocessor",
    "ConcatMixer", "WeightedInterleaveMixer",
    "CrossEntropyLoss", "CEPlusOrthoLoss",
]
