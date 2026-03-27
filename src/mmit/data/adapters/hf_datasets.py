"""HuggingFace Datasets adapter — load any HF Hub dataset as CanonicalSamples.

Supports two common dataset formats:
* **conversations** — LLaVA-style ``conversations`` column
* **vqa** — flat question/answer columns with optional PIL image column

Auto-detection tries known dataset mappings first, then inspects column names.

Example
-------
>>> from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
>>> adapter = HFDatasetsAdapter("liuhaotian/LLaVA-Instruct-150K", split="train", max_samples=10)
>>> for sample in adapter:
...     print(sample.id, sample.first_question[:60])
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from mmit.data.adapters.base import DatasetAdapter
from mmit.data.types import CanonicalSample, Turn


# ---------------------------------------------------------------------------
# Known dataset column mappings
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    """Describes how columns in a HF dataset map to CanonicalSample fields."""
    format: str = "vqa"         # "conversations" | "vqa"
    id_col: str = "id"
    image_col: str = "image"    # may contain path (str) or PIL Image
    question_col: str = "question"
    answer_col: str = "answer"
    conversations_col: str = "conversations"


@dataclass
class DatasetProfile:
    """Column mapping + evaluation instruction config for a HF dataset.

    ``task_type`` controls how the instruction prompt is assembled:
    - ``open_vqa``        — raw question + instruction_suffix
    - ``mcq``             — question + formatted choices + instruction_suffix
    - ``caption``         — instruction_suffix used as the prompt (question ignored)
    - ``yes_no``          — raw question + instruction_suffix
    - ``classification``  — instruction_suffix as prompt, label → class name
    """
    columns: ColumnMapping
    task_type: str = "open_vqa"          # "open_vqa" | "mcq" | "caption" | "yes_no" | "classification"
    instruction_suffix: str = ""         # appended after the question
    choices_col: str = ""                # column containing answer choices (list)
    hint_col: str = ""                   # column containing context / hint text
    answer_key_col: str = ""             # MCQ correct answer index column
    config_name: str = ""                # HF dataset config name (e.g. "ScienceQA-IMG")
    image_config_name: str = ""          # separate config for images (GQA-style dual-config)
    image_join_key: str = ""             # column in instructions linking to image config's 'id'


# ---------------------------------------------------------------------------
# Instruction builder
# ---------------------------------------------------------------------------

def _build_instruction(
    question: str,
    profile: DatasetProfile,
    row: dict,
) -> str:
    """Assemble a complete inference prompt from *question* + profile config.

    For ``caption`` tasks where no question exists, the suffix itself is the
    prompt.  For ``mcq`` tasks, choices are formatted as ``A. … / B. …``
    before the suffix.
    """
    if profile.task_type == "caption":
        return profile.instruction_suffix or question

    parts: List[str] = []

    # Context / hint (ScienceQA, MMBench, …)
    if profile.hint_col:
        hint = str(row.get(profile.hint_col, "")).strip()
        if hint:
            parts.append(f"Context: {hint}")

    parts.append(question)

    # Multiple-choice options
    if profile.task_type == "mcq" and profile.choices_col:
        choices = row.get(profile.choices_col, [])
        if isinstance(choices, list) and choices:
            parts.append(
                "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            )

    if profile.instruction_suffix:
        parts.append(profile.instruction_suffix)

    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Pre-defined profiles for popular datasets
# ---------------------------------------------------------------------------

_SHORT_ANSWER = "Answer the question using a single word or phrase."
_MCQ_LETTER = "Answer with the option's letter from the given choices directly."

DATASET_PROFILES: Dict[str, DatasetProfile] = {
    # --- 对话格式（训练集）---
    "liuhaotian/LLaVA-Instruct-150K": DatasetProfile(
        columns=ColumnMapping(
            format="conversations", id_col="id", image_col="image",
            conversations_col="conversations",
        ),
    ),
    "liuhaotian/LLaVA-CC3M-Pretrain-595K": DatasetProfile(
        columns=ColumnMapping(
            format="conversations", id_col="id", image_col="image",
            conversations_col="conversations",
        ),
    ),
    # --- Open VQA ---
    "merve/vqav2-small": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="", image_col="image",
            question_col="question", answer_col="multiple_choice_answer",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
    ),
    "flaviagiammarino/path-vqa": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="", image_col="image",
            question_col="question", answer_col="answer",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
    ),
    "lmms-lab/textvqa": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="question_id", image_col="image",
            question_col="question", answer_col="answers",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
    ),
    "howard-hou/OCR-VQA": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="image_id", image_col="image",
            question_col="questions", answer_col="answers",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
    ),
    # --- Caption ---
    "lmms-lab/flickr30k": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="img_id", image_col="image",
            question_col="caption", answer_col="",
        ),
        task_type="caption",
        instruction_suffix="Describe this image.",
    ),
    # --- Multiple-choice ---
    "Gregor/scienceqa": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="id", image_col="image",
            question_col="question", answer_col="answer",
        ),
        task_type="mcq",
        instruction_suffix=_MCQ_LETTER,
        choices_col="choices",
        hint_col="hint",
        answer_key_col="answer",
    ),
    # --- lmms-lab ScienceQA (image subset, val/test only) ---
    "lmms-lab/ScienceQA": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="", image_col="image",
            question_col="question", answer_col="answer",
        ),
        task_type="mcq",
        instruction_suffix=_MCQ_LETTER,
        choices_col="choices",
        hint_col="hint",
        answer_key_col="answer",
        config_name="ScienceQA-IMG",
    ),
    # --- VQAv2 (lmms-lab full version) ---
    "lmms-lab/VQAv2": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="question_id", image_col="image",
            question_col="question", answer_col="multiple_choice_answer",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
    ),
    # --- GQA (dual-config: instructions + images) ---
    "lmms-lab/GQA": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="id", image_col="",
            question_col="question", answer_col="answer",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
        config_name="testdev_balanced_instructions",
        image_config_name="testdev_balanced_images",
        image_join_key="imageId",
    ),
    # --- VizWiz VQA ---
    "lmms-lab/VizWiz-VQA": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="question_id", image_col="image",
            question_col="question", answer_col="answers",
        ),
        task_type="open_vqa",
        instruction_suffix=_SHORT_ANSWER,
    ),
    # --- Grounding (RefCOCO) ---
    "lmms-lab/RefCOCO": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="question_id", image_col="image",
            question_col="question", answer_col="answer",
        ),
        task_type="open_vqa",
        instruction_suffix="Describe the object in the highlighted region briefly.",
    ),
    # --- ImageNet-1k (classification, gated) ---
    "ILSVRC/imagenet-1k": DatasetProfile(
        columns=ColumnMapping(
            format="vqa", id_col="", image_col="image",
            question_col="", answer_col="label",
        ),
        task_type="classification",
        instruction_suffix="What is the main object in this image? Answer with a single word or phrase.",
    ),
}

# Backward-compatible alias used by HFDatasetsAdapter
KNOWN_DATASET_MAPPINGS: Dict[str, ColumnMapping] = {
    name: profile.columns for name, profile in DATASET_PROFILES.items()
}

# Datasets that only have image filenames (not embedded PIL images).
# These are typically training datasets that reference images from COCO etc.
TEXT_ONLY_DATASETS: set = {
    "liuhaotian/LLaVA-Instruct-150K",
    "liuhaotian/LLaVA-CC3M-Pretrain-595K",
}


# Datasets known to fail or be very slow with non-streaming load
_PREFER_STREAMING: set = {
    "liuhaotian/LLaVA-Instruct-150K",
    "liuhaotian/LLaVA-CC3M-Pretrain-595K",
    "howard-hou/OCR-VQA",
    "ILSVRC/imagenet-1k",
    "lmms-lab/VQAv2",
}


def _detect_column_mapping(dataset) -> ColumnMapping:
    """Auto-detect column mapping from dataset column names."""
    cols = set(dataset.column_names)

    # Check for conversations format
    if "conversations" in cols:
        return ColumnMapping(
            format="conversations",
            id_col="id" if "id" in cols else "",
            image_col="image" if "image" in cols else "",
            conversations_col="conversations",
        )

    # VQA-style
    mapping = ColumnMapping(format="vqa")

    # ID column
    for candidate in ("id", "question_id", "sample_id", "idx"):
        if candidate in cols:
            mapping.id_col = candidate
            break

    # Image column
    for candidate in ("image", "image_path", "img", "img_path", "file_name"):
        if candidate in cols:
            mapping.image_col = candidate
            break

    # Question column
    for candidate in ("question", "text_input", "input_text", "prompt", "instruction"):
        if candidate in cols:
            mapping.question_col = candidate
            break

    # Answer column
    for candidate in ("answer", "answers", "multiple_choice_answer", "output_text",
                       "response", "caption", "label"):
        if candidate in cols:
            mapping.answer_col = candidate
            break

    return mapping


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_conversations_row(row: dict, idx: int, mapping: ColumnMapping,
                             load_images: bool = True) -> CanonicalSample:
    """Parse a HF dataset row with conversations format."""
    convs = row.get(mapping.conversations_col, [])
    turns: List[Turn] = []
    for msg in convs:
        if isinstance(msg, dict):
            role_raw = str(msg.get("from", msg.get("role", "human"))).lower()
            role = "human" if role_raw in ("human", "user") else "assistant"
            content = str(msg.get("value", msg.get("content", ""))).strip()
            # Strip image tokens
            if role == "human":
                content = content.replace("<image>", "").replace("<Image>", "").strip()
            if content:
                turns.append(Turn(role=role, content=content))

    # Handle image: could be PIL Image or string path
    image_val = row.get(mapping.image_col)
    image_path, metadata = _handle_image_value(image_val, load_images)

    sample_id = str(row.get(mapping.id_col, idx)) if mapping.id_col else str(idx)

    return CanonicalSample(
        id=sample_id,
        image_path=image_path,
        turns=turns,
        metadata=metadata,
    )


def _parse_vqa_row(row: dict, idx: int, mapping: ColumnMapping,
                   load_images: bool = True,
                   profile: Optional[DatasetProfile] = None) -> CanonicalSample:
    """Parse a HF dataset row with VQA format."""
    question_raw = row.get(mapping.question_col, "") if mapping.question_col else ""
    # Handle list-valued question fields (e.g. flickr30k captions, OCR-VQA questions)
    if isinstance(question_raw, list):
        question = str(question_raw[0]).strip() if question_raw else ""
    else:
        question = str(question_raw).strip()
    answer_raw = row.get(mapping.answer_col, "") if mapping.answer_col else ""

    # Handle list of answers (e.g. VQAv2 'answers' field)
    if isinstance(answer_raw, list):
        if answer_raw and isinstance(answer_raw[0], dict):
            answer = answer_raw[0].get("answer", str(answer_raw[0]))
        else:
            answer = str(answer_raw[0]) if answer_raw else ""
    else:
        answer = str(answer_raw).strip()

    # For MCQ datasets, convert answer index to letter (e.g. 2 → "C")
    if profile and profile.task_type == "mcq" and profile.answer_key_col:
        try:
            answer_idx = int(row.get(profile.answer_key_col, -1))
            if 0 <= answer_idx < 26:
                answer = chr(65 + answer_idx)
        except (ValueError, TypeError):
            pass

    turns: List[Turn] = []
    if question:
        turns.append(Turn(role="human", content=question))
    if answer:
        turns.append(Turn(role="assistant", content=answer))

    # Handle image
    image_val = row.get(mapping.image_col)
    image_path, metadata = _handle_image_value(image_val, load_images)

    # Store raw answer(s) for evaluation scoring
    if isinstance(answer_raw, list):
        if answer_raw and isinstance(answer_raw[0], dict):
            metadata["raw_answers"] = [a.get("answer", str(a)) for a in answer_raw]
        else:
            metadata["raw_answers"] = [str(a) for a in answer_raw]
    elif answer_raw:
        metadata["raw_answers"] = [str(answer_raw).strip()]

    # Store task_type if known
    if profile:
        metadata["task_type"] = profile.task_type

    sample_id = str(row.get(mapping.id_col, idx)) if mapping.id_col else str(idx)

    # Build formatted instruction from profile
    instruction = ""
    if profile and profile.instruction_suffix:
        instruction = _build_instruction(question, profile, row)

    return CanonicalSample(
        id=sample_id,
        image_path=image_path,
        turns=turns,
        metadata=metadata,
        instruction=instruction,
    )


def _handle_image_value(image_val, load_images: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Handle image column value: could be str path, PIL Image, or undecoded dict."""
    metadata: Dict[str, Any] = {}
    image_path = ""

    if image_val is None:
        pass
    elif isinstance(image_val, str):
        image_path = image_val
    elif isinstance(image_val, dict) and ("bytes" in image_val or "path" in image_val):
        # Undecoded image from datasets library (Image(decode=False))
        if load_images and image_val.get("bytes"):
            try:
                import io
                from PIL import Image
                pil_img = Image.open(io.BytesIO(image_val["bytes"]))
                metadata["_pil_image"] = pil_img
                image_path = "<in_memory>"
            except Exception:
                image_path = image_val.get("path", "<deferred>")
        else:
            image_path = image_val.get("path", "<deferred>")
            if image_val.get("bytes"):
                metadata["_image_bytes"] = image_val["bytes"]
    else:
        if load_images:
            # Assume PIL Image (HF datasets often return Image objects)
            try:
                from PIL import Image
                if isinstance(image_val, Image.Image):
                    metadata["_pil_image"] = image_val
                    image_path = "<in_memory>"
            except ImportError:
                pass
        else:
            # Store reference for lazy decoding
            metadata["_raw_image"] = image_val
            image_path = "<deferred>"

    return image_path, metadata


def decode_sample_image(sample) -> Optional[Any]:
    """Decode a deferred/lazy image from a CanonicalSample on demand.

    Returns a PIL Image or None.
    """
    # Already decoded
    pil = sample.metadata.get("_pil_image")
    if pil is not None:
        return pil

    # Raw PIL Image stored as deferred (load_images=False)
    raw = sample.metadata.get("_raw_image")
    if raw is not None:
        try:
            from PIL import Image
            if isinstance(raw, Image.Image):
                return raw
        except ImportError:
            pass

    # Bytes from undecoded HF Image feature
    img_bytes = sample.metadata.get("_image_bytes")
    if img_bytes is not None:
        try:
            import io
            from PIL import Image
            return Image.open(io.BytesIO(img_bytes))
        except Exception:
            pass

    # String path on disk
    if sample.image_path and sample.image_path not in ("", "<deferred>", "<in_memory>"):
        try:
            from PIL import Image
            return Image.open(sample.image_path)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# HFDatasetsAdapter
# ---------------------------------------------------------------------------

class HFDatasetsAdapter(DatasetAdapter):
    """Load a HuggingFace Hub dataset and yield CanonicalSamples.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset id (e.g. ``"liuhaotian/LLaVA-Instruct-150K"``).
    split:
        Dataset split to load (default ``"train"``).
    column_map:
        Optional :class:`ColumnMapping` override. Auto-detected if ``None``.
    max_samples:
        Limit number of samples loaded. ``None`` for full dataset.
    streaming:
        Use streaming mode for large datasets. When ``True``, ``__len__``
        returns ``max_samples`` (or -1 if unlimited).
    trust_remote_code:
        Passed to ``datasets.load_dataset()``.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        column_map: Optional[ColumnMapping] = None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
        trust_remote_code: bool = True,
        load_images: bool = True,
        config_name: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.streaming = streaming
        self.load_images = load_images
        self._image_lookup: Dict[str, Any] = {}
        self._label_names: Optional[List[str]] = None

        # Early profile lookup — needed to determine config_name before loading
        _profile_early = DATASET_PROFILES.get(dataset_name)
        _config = config_name or (_profile_early.config_name if _profile_early else "")

        # Helper: build positional args for load_dataset / load_dataset_builder
        _load_pos = (dataset_name, _config) if _config else (dataset_name,)

        # Load dataset
        try:
            import datasets
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HFDatasetsAdapter. "
                "Install it with: pip install datasets"
            )

        # Auto-enable streaming for known slow datasets or small sample requests
        # (avoids downloading multi-GB datasets just for a few samples)
        if not streaming and (
            dataset_name in _PREFER_STREAMING
            or (max_samples is not None and max_samples <= 50)
        ):
            streaming = True
            self.streaming = True

        # Build list of splits to try (requested split first, then fallbacks)
        splits_to_try = [split]
        try:
            ds_info = datasets.load_dataset_builder(*_load_pos).info
            available = list(ds_info.splits.keys()) if ds_info.splits else []
            if available:
                if split in available:
                    splits_to_try = [split]
                else:
                    # Requested split not available — use available ones
                    splits_to_try = available
                    self.split = splits_to_try[0]
        except Exception:
            pass
        # Always append common fallback splits so we don't fail silently
        for fallback in ("train", "validation", "test"):
            if fallback not in splits_to_try:
                splits_to_try.append(fallback)

        # Try loading with multiple fallback strategies
        first_err = None
        loaded = False

        for try_split in splits_to_try:
            load_kwargs: Dict[str, Any] = {"split": try_split, "streaming": streaming}
            for kwargs in [load_kwargs, {**load_kwargs, "trust_remote_code": True}]:
                try:
                    self._hf_dataset = datasets.load_dataset(*_load_pos, **kwargs)
                    self.split = try_split
                    loaded = True
                    break
                except Exception as e:
                    if first_err is None:
                        first_err = e
            if loaded:
                break

        # Streaming fallback for datasets with pyarrow / schema issues
        if not loaded and not streaming:
            for try_split in splits_to_try:
                for kwargs in [
                    {"split": try_split, "streaming": True},
                    {"split": try_split, "streaming": True, "trust_remote_code": True},
                ]:
                    try:
                        self._hf_dataset = datasets.load_dataset(*_load_pos, **kwargs)
                        self.streaming = True
                        self.split = try_split
                        loaded = True
                        break
                    except Exception:
                        pass
                if loaded:
                    break

        if not loaded:
            raise first_err  # type: ignore[misc]

        # For non-streaming datasets, optionally disable eager image decoding
        if not self.load_images and not self.streaming:
            try:
                for col_name in self._hf_dataset.column_names:
                    feat = self._hf_dataset.features.get(col_name)
                    if isinstance(feat, datasets.Image):
                        self._hf_dataset = self._hf_dataset.cast_column(
                            col_name, datasets.Image(decode=False)
                        )
            except Exception:
                pass

        if max_samples is not None and not self.streaming:
            self._hf_dataset = self._hf_dataset.select(
                range(min(max_samples, len(self._hf_dataset)))
            )

        # Determine column mapping + dataset profile
        if dataset_name in DATASET_PROFILES:
            self._profile = DATASET_PROFILES[dataset_name]
            self._mapping = column_map or self._profile.columns
        else:
            self._profile = None
            if column_map is not None:
                self._mapping = column_map
            else:
                self._mapping = _detect_column_mapping(self._hf_dataset)

        # --- Post-load: join images from separate config (GQA-style) ---
        if (self._profile and self._profile.image_config_name
                and self._profile.image_join_key and self.load_images):
            self._join_images_from_config(datasets, _profile_early)

        # --- Post-load: extract label names for classification ---
        if not self.streaming:
            answer_col = self._mapping.answer_col
            if answer_col:
                try:
                    feat = self._hf_dataset.features.get(answer_col)
                    if hasattr(feat, "names"):
                        self._label_names = feat.names
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _join_images_from_config(self, datasets_mod, profile: DatasetProfile):
        """Load images from a separate HF config and build a lookup dict."""
        try:
            join_key = profile.image_join_key
            # Collect needed image IDs from the loaded instructions
            needed_ids: set = set()
            for row in self._hf_dataset:
                img_id = row.get(join_key, "")
                if img_id:
                    needed_ids.add(str(img_id))
            if not needed_ids:
                return

            # Load image config in streaming mode to avoid downloading everything
            img_ds = datasets_mod.load_dataset(
                self.dataset_name, profile.image_config_name,
                split=self.split, streaming=True,
            )
            for row in img_ds:
                img_id = str(row.get("id", ""))
                if img_id in needed_ids:
                    self._image_lookup[img_id] = row.get("image")
                if len(self._image_lookup) >= len(needed_ids):
                    break
        except Exception:
            pass  # fail silently — images just won't be available

    def _postprocess_sample(self, sample: CanonicalSample, row: dict) -> CanonicalSample:
        """Inject joined images and convert classification labels."""
        # Inject images from lookup (GQA-style dual-config)
        if self._image_lookup and self._profile and self._profile.image_join_key:
            img_id = str(row.get(self._profile.image_join_key, ""))
            if img_id in self._image_lookup:
                img_val = self._image_lookup[img_id]
                if img_val is not None:
                    try:
                        from PIL import Image as _PILImage
                        if isinstance(img_val, _PILImage.Image):
                            sample.metadata["_pil_image"] = img_val
                            sample.image_path = "<in_memory>"
                    except ImportError:
                        pass

        # Convert classification labels (e.g. ImageNet int → class name)
        if self._label_names:
            for turn in sample.turns:
                if turn.role == "assistant" and turn.content.lstrip("-").isdigit():
                    label_idx = int(turn.content)
                    if 0 <= label_idx < len(self._label_names):
                        turn.content = self._label_names[label_idx].split(",")[0].strip()
            raw = sample.metadata.get("raw_answers", [])
            if raw:
                new_raw = []
                for a in raw:
                    if a.lstrip("-").isdigit():
                        idx = int(a)
                        if 0 <= idx < len(self._label_names):
                            new_raw.append(self._label_names[idx].split(",")[0].strip())
                        else:
                            new_raw.append(a)
                    else:
                        new_raw.append(a)
                sample.metadata["raw_answers"] = new_raw

        return sample

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self.streaming:
            return self.max_samples if self.max_samples is not None else -1
        return len(self._hf_dataset)

    def __iter__(self) -> Iterator[CanonicalSample]:
        is_conv = self._mapping.format == "conversations"
        need_postprocess = bool(self._image_lookup or self._label_names)
        count = 0
        for idx, row in enumerate(self._hf_dataset):
            if self.max_samples is not None and count >= self.max_samples:
                break
            if is_conv:
                sample = _parse_conversations_row(row, idx, self._mapping, self.load_images)
            else:
                sample = _parse_vqa_row(row, idx, self._mapping, self.load_images,
                                        profile=self._profile)
            if need_postprocess:
                sample = self._postprocess_sample(sample, row)
            yield sample
            count += 1

    def __getitem__(self, idx: int) -> CanonicalSample:
        if self.streaming:
            raise TypeError("__getitem__ is not supported in streaming mode.")
        row = self._hf_dataset[idx]
        if self._mapping.format == "conversations":
            sample = _parse_conversations_row(row, idx, self._mapping, self.load_images)
        else:
            sample = _parse_vqa_row(row, idx, self._mapping, self.load_images,
                                    profile=self._profile)
        if self._image_lookup or self._label_names:
            sample = self._postprocess_sample(sample, row)
        return sample

    @property
    def column_names(self) -> List[str]:
        """Return the column names of the underlying HF dataset."""
        return self._hf_dataset.column_names

    @property
    def mapping(self) -> ColumnMapping:
        """Return the resolved column mapping."""
        return self._mapping

    @property
    def profile(self) -> Optional[DatasetProfile]:
        """Return the resolved dataset profile (None for unknown datasets)."""
        return self._profile
