"""MultiImagePreprocessor — handles samples with multiple images.

For papers like CaD-VI that compare two images side-by-side.
Extra images are stored in sample.metadata["extra_images"] as a list of paths.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PIL import Image

from mmit.data.types import CanonicalSample
from mmit.training.preprocessors.base import IGNORE_INDEX
from mmit.training.preprocessors.chat_template import (
    ChatTemplatePreprocessor,
    _load_image,
)


class MultiImagePreprocessor(ChatTemplatePreprocessor):
    """Extends ChatTemplatePreprocessor to handle multiple images per sample."""

    def supports_multi_image(self) -> bool:
        return True

    def tokenize(
        self,
        sample: CanonicalSample,
        processor: Any,
        image_root: str = "",
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        # Collect all images
        images: List[Image.Image] = []
        primary = _load_image(sample, image_root)
        if primary is not None:
            images.append(primary)

        extra_paths = (sample.metadata or {}).get("extra_images", [])
        for path in extra_paths:
            full_path = os.path.join(image_root, path) if image_root else path
            if os.path.isfile(full_path):
                images.append(Image.open(full_path).convert("RGB"))

        if not images:
            # Fall back to text-only
            return super().tokenize(sample, processor, image_root, max_length)

        # Build messages with multiple image tokens
        messages = []
        for turn in sample.turns:
            role = "user" if turn.role == "human" else "assistant"
            if role == "user" and not messages:
                content = []
                for _ in images:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": turn.content})
            else:
                content = [{"type": "text", "text": turn.content}]
            messages.append({"role": role, "content": content})

        # Full text
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # Prompt-only text for label masking
        last_asst_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_asst_idx = i
                break
        prompt_messages = messages[:last_asst_idx] if last_asst_idx > 0 else []
        prompt_text = ""
        if prompt_messages:
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

        # Tokenize
        full_inputs = processor(
            text=full_text, images=images,
            return_tensors="pt", truncation=True, max_length=max_length,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        if prompt_text:
            prompt_inputs = processor(
                text=prompt_text, images=images,
                return_tensors="pt", truncation=True, max_length=max_length,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[:prompt_len] = IGNORE_INDEX

        import torch
        attention_mask = full_inputs.get("attention_mask", torch.ones_like(input_ids))
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.squeeze(0)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if "pixel_values" in full_inputs:
            pv = full_inputs["pixel_values"]
            result["pixel_values"] = pv.squeeze(0) if pv.dim() == 5 else pv
        if "image_sizes" in full_inputs:
            result["image_sizes"] = full_inputs["image_sizes"]
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"]

        return result
