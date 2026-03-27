"""ChatTemplatePreprocessor — universal preprocessor using HF chat templates.

Works for any VLM whose processor supports ``apply_chat_template``:
LLaVA, Qwen2-VL, Qwen2.5-VL, Gemma 3, etc.

Label masking strategy (two-pass diff):
  1. Build the full conversation text (user + assistant turns).
  2. Build the prompt-only text (everything up to the assistant response).
  3. Tokenize both; mask labels for all tokens in the prompt-only portion.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from mmit.data.types import CanonicalSample
from mmit.training.preprocessors.base import IGNORE_INDEX, Preprocessor


def _load_image(sample: CanonicalSample, image_root: str = "") -> Optional[Image.Image]:
    """Load a PIL Image from a CanonicalSample."""
    if not sample.image_path:
        return None
    pil_image = sample.metadata.get("_pil_image") if sample.metadata else None
    if pil_image is not None:
        return pil_image.convert("RGB")
    img_path = (
        os.path.join(image_root, sample.image_path)
        if image_root
        else sample.image_path
    )
    if not os.path.isfile(img_path):
        return None
    return Image.open(img_path).convert("RGB")


def _build_messages(sample: CanonicalSample, has_image: bool) -> List[Dict]:
    """Convert CanonicalSample turns into HF chat messages format."""
    messages = []
    for turn in sample.turns:
        role = "user" if turn.role == "human" else "assistant"
        if role == "user" and has_image and not messages:
            # First user turn with image
            content = [
                {"type": "image"},
                {"type": "text", "text": turn.content},
            ]
        else:
            content = [{"type": "text", "text": turn.content}]
        messages.append({"role": role, "content": content})
    return messages


def _build_prompt_messages(messages: List[Dict]) -> List[Dict]:
    """Build messages up to (but excluding) the last assistant response.

    Used for the two-pass diff label masking strategy.
    """
    # Find the last assistant message
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx < 0:
        return messages
    return messages[:last_assistant_idx]


class ChatTemplatePreprocessor(Preprocessor):
    """Preprocessor using HF processor.apply_chat_template().

    Handles single-image, single/multi-turn conversations.
    """

    def tokenize(
        self,
        sample: CanonicalSample,
        processor: Any,
        image_root: str = "",
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        image = _load_image(sample, image_root)
        has_image = image is not None
        messages = _build_messages(sample, has_image)

        if not messages:
            raise ValueError(f"Sample {sample.id} has no turns")

        # Full conversation text
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # Prompt-only text (for label masking)
        prompt_messages = _build_prompt_messages(messages)
        if prompt_messages:
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = ""

        # Tokenize full conversation
        images = [image] if has_image else None
        full_inputs = processor(
            text=full_text, images=images,
            return_tensors="pt", truncation=True, max_length=max_length,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)

        # Build labels: mask prompt tokens with IGNORE_INDEX
        labels = input_ids.clone()
        if prompt_text:
            prompt_inputs = processor(
                text=prompt_text, images=images,
                return_tensors="pt", truncation=True, max_length=max_length,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[:prompt_len] = IGNORE_INDEX

        attention_mask = full_inputs.get("attention_mask", torch.ones_like(input_ids))
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.squeeze(0)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Pass through vision-specific fields
        # Processor may output (batch, num_images, C, H, W) — squeeze to (C, H, W) per sample
        if "pixel_values" in full_inputs:
            pv = full_inputs["pixel_values"]
            while pv.dim() > 3 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            result["pixel_values"] = pv
        if "image_sizes" in full_inputs:
            result["image_sizes"] = full_inputs["image_sizes"]
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"]

        return result

    def collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            return {}

        # Pad input_ids, labels, attention_mask
        max_len = max(s["input_ids"].size(0) for s in samples)
        batch_size = len(samples)

        batch_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        batch_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, s in enumerate(samples):
            seq_len = s["input_ids"].size(0)
            batch_ids[i, :seq_len] = s["input_ids"]
            batch_labels[i, :seq_len] = s["labels"]
            batch_mask[i, :seq_len] = s["attention_mask"]

        batch: Dict[str, Any] = {
            "input_ids": batch_ids,
            "labels": batch_labels,
            "attention_mask": batch_mask,
        }

        # Handle pixel_values: stack if same shape, concat if variable
        if "pixel_values" in samples[0]:
            pvs = [s["pixel_values"] for s in samples]
            try:
                batch["pixel_values"] = torch.stack(pvs)
            except RuntimeError:
                # Variable shape (e.g. Qwen2-VL) — concat along batch dim
                batch["pixel_values"] = torch.cat(pvs, dim=0)

        # Pass through list-type fields
        for key in ("image_sizes", "image_grid_thw"):
            if key in samples[0]:
                vals = [s[key] for s in samples]
                if isinstance(vals[0], torch.Tensor):
                    try:
                        batch[key] = torch.cat(vals, dim=0)
                    except RuntimeError:
                        batch[key] = vals
                else:
                    batch[key] = vals

        return batch
