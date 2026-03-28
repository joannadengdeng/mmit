"""LocalMethod — run inference with a locally loaded model (base or PEFT checkpoint).

Used for evaluating trained models on benchmarks like VQAv2, POPE, TextVQA.

Usage:
    method = LocalMethod.from_checkpoint(
        base_model_id="llava-hf/llava-1.5-7b-hf",
        checkpoint_path="output/qlora/final",
        ft_method="qlora",
    )
    # Or just a base model:
    method = LocalMethod(model, processor)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from PIL import Image

from mmit.data.types import CanonicalSample, EvalSample
from mmit.eval.methods.base import Method


class LocalMethod(Method):
    """Inference with a locally loaded VLM model.

    Optimized for eval: short max_new_tokens, greedy decoding.
    """

    def __init__(self, model, processor, device=None):
        self.model = model
        self.processor = processor
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        base_model_id: str,
        checkpoint_path: str = "",
        ft_method: str = "",
        quantize_4bit: bool = True,
        **kwargs,
    ) -> "LocalMethod":
        """Load a base model + optional PEFT checkpoint."""
        import json
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

        if quantize_4bit:
            model = AutoVLM.from_pretrained(
                base_model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                device_map="auto", trust_remote_code=True,
            )
        else:
            model = AutoVLM.from_pretrained(
                base_model_id, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )

        if checkpoint_path and os.path.isdir(checkpoint_path):
            if not ft_method:
                meta_path = os.path.join(checkpoint_path, "mmit_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    ft_method = meta.get("ft_method", "")

            if ft_method in ("qlora", "lora", "dora", "l2t", "lora_in_lora"):
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, checkpoint_path)
                try:
                    model = model.merge_and_unload()
                except Exception:
                    pass
            elif ft_method == "freeze":
                state = torch.load(
                    os.path.join(checkpoint_path, "freeze_tuned.pt"),
                    map_location="cpu", weights_only=True,
                )
                model.load_state_dict(state, strict=False)

        model.eval()
        return cls(model, processor)

    def prepare_input(
        self,
        sample: CanonicalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Prepare input for generation."""
        image = None
        if sample.image_path:
            pil = (sample.metadata or {}).get("_pil_image")
            if pil is not None:
                image = pil.convert("RGB")
            else:
                img_path = os.path.join(image_root, sample.image_path) if image_root else sample.image_path
                if os.path.isfile(img_path):
                    image = Image.open(img_path).convert("RGB")

        question = ""
        for turn in sample.turns:
            if turn.role == "human":
                question = turn.content
                break
        if not question:
            question = "Describe this image."

        if image is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images = [image] if image is not None else None
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response. Default max_new_tokens=32 for short VQA answers."""
        with torch.no_grad():
            output = self.model.generate(
                **prepared,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = prepared["input_ids"].shape[1]
        response = self.processor.decode(
            output[0][prompt_len:], skip_special_tokens=True,
        )
        return response.strip()
