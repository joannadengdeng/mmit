"""LLaVA JSON / JSONL dataset adapter.

Handles both the standard ``conversations`` format and simpler
``input_text`` / ``output_text`` flat format.

Supported on-disk schemas
--------------------------
1. LLaVA native (conversations array)::

    {
        "id": "000000033471",
        "image": "train2017/000000033471.jpg",
        "conversations": [
            {"from": "human",     "value": "<image>\\nWhat is this?"},
            {"from": "gpt",       "value": "A dog."},
            {"from": "human",     "value": "What breed?"},
            {"from": "gpt",       "value": "Labrador."}
        ]
    }

2. Flat format::

    {
        "id": "123",
        "image": "img.jpg",
        "input_text": "Describe the image.",
        "output_text": "A sunny beach."
    }
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List

from mmit.data.adapters.base import DatasetAdapter
from mmit.data.types import CanonicalSample, Turn

# Tokens to strip from human messages when normalising
_IMAGE_TOKENS = {"<image>", "<Image>"}


def _strip_image_token(text: str) -> str:
    for tok in _IMAGE_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def _parse_conversations(raw: dict, idx: int) -> CanonicalSample:
    """Parse a record that has a ``conversations`` field."""
    turns: List[Turn] = []
    for msg in raw.get("conversations", []):
        role_raw = str(msg.get("from", "human")).lower()
        role = "human" if role_raw in {"human", "user"} else "assistant"
        content = str(msg.get("value", "")).strip()
        if not content:
            continue
        if role == "human":
            content = _strip_image_token(content)
        turns.append(Turn(role=role, content=content))

    return CanonicalSample(
        id=str(raw.get("id", idx)),
        image_path=str(raw.get("image", "")),
        turns=turns,
        metadata={k: v for k, v in raw.items() if k not in {"id", "image", "conversations"}},
    )


def _parse_flat(raw: dict, idx: int) -> CanonicalSample:
    """Parse a flat record without ``conversations``."""
    input_text = str(raw.get("input_text", raw.get("question", raw.get("text_input", "")))).strip()
    output_text = str(raw.get("output_text", raw.get("answer", raw.get("caption", "")))).strip()
    turns: List[Turn] = []
    if input_text:
        turns.append(Turn(role="human", content=input_text))
    if output_text:
        turns.append(Turn(role="assistant", content=output_text))
    return CanonicalSample(
        id=str(raw.get("id", idx)),
        image_path=str(raw.get("image", "")),
        turns=turns,
        metadata={k: v for k, v in raw.items()
                  if k not in {"id", "image", "input_text", "output_text",
                                "question", "answer", "caption"}},
    )


def _parse_record(raw: dict, idx: int) -> CanonicalSample:
    if "conversations" in raw:
        return _parse_conversations(raw, idx)
    return _parse_flat(raw, idx)


class LLaVAJSONAdapter(DatasetAdapter):
    """Load a LLaVA-format JSON or JSONL file.

    Parameters
    ----------
    data_path:
        Path to the ``.json`` (list) or ``.jsonl`` (one record per line) file.
    image_root:
        Directory prepended to each sample's ``image_path`` when resolving
        absolute paths.  The adapter itself stores relative paths; callers
        resolve them at access time.
    """

    def __init__(self, data_path: str, image_root: str = "") -> None:
        self.data_path = data_path
        self.image_root = image_root
        self._records = self._load(data_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: str) -> List[dict]:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CanonicalSample]:
        for idx, raw in enumerate(self._records):
            yield _parse_record(raw, idx)

    def __getitem__(self, idx: int) -> CanonicalSample:
        return _parse_record(self._records[idx], idx)
