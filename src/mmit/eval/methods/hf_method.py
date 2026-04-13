"""HFMethod — unified HuggingFace multimodal inference wrapper.

Calls HuggingFace Inference API via ``InferenceClient`` (serverless, no GPU needed).

Supported model families
------------------------
* ``qwen2_5_vl``     — Qwen2.5-VL (recommended, has inference providers)
* ``qwen3_vl``       — Qwen3-VL
* ``gemma3``         — Gemma 3 / Gemma 3n
* ``llama4``         — Llama 4 Scout
* ``glm4v``          — GLM-4.5V / GLM-4.6V
* ``llava``          — LLaVA 1.5 / Interleave (no inference provider)
* ``llava_next``     — LLaVA-NeXT v1.6 (no inference provider)
* ``qwen2_vl``       — Qwen2-VL (no inference provider)
* ``paligemma``      — PaliGemma (no inference provider)
* ``instructblip``   — InstructBLIP (no inference provider)

Example (API mode)
------------------
>>> method = HFMethod("Qwen/Qwen2.5-VL-7B-Instruct")
>>> result = method.inference("What is in this image?", "path/to/image.jpg")
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from mmit.data.types import CanonicalSample, EvalSample, Turn
from mmit.eval.methods.base import Method


# ---------------------------------------------------------------------------
# Model family configuration
# ---------------------------------------------------------------------------

@dataclass
class FamilyConfig:
    """Description of one HuggingFace VLM family."""
    auto_class: str                      # e.g. "LlavaForConditionalGeneration"
    supports_chat_template: bool = True  # processor.apply_chat_template available?
    image_token: str = "<image>"         # placeholder in prompt
    default_models: List[str] = field(default_factory=list)
    # Whether the model supports chat_completion API (multimodal chat)
    supports_chat_api: bool = True


MODEL_FAMILY_CONFIGS: Dict[str, FamilyConfig] = {
    # ---- Models WITH inference providers (recommended) ----
    "qwen2_5_vl": FamilyConfig(
        auto_class="Qwen2_5_VLForConditionalGeneration",
        supports_chat_template=True,
        image_token="<|image_pad|>",
        supports_chat_api=True,
        default_models=[
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ],
    ),
    "qwen3_vl": FamilyConfig(
        auto_class="Qwen3VLForConditionalGeneration",
        supports_chat_template=True,
        image_token="<|image_pad|>",
        supports_chat_api=True,
        default_models=[
            "Qwen/Qwen3-VL-8B-Instruct",
        ],
    ),
    "gemma3": FamilyConfig(
        auto_class="Gemma3ForConditionalGeneration",
        supports_chat_template=True,
        image_token="",
        supports_chat_api=True,
        default_models=[
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
            "google/gemma-3n-E4B-it",
        ],
    ),
    "mllama": FamilyConfig(
        auto_class="MllamaForConditionalGeneration",
        supports_chat_template=True,
        image_token="",
        supports_chat_api=True,
        default_models=[
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct",
        ],
    ),
    "llama4": FamilyConfig(
        auto_class="Llama4ForConditionalGeneration",
        supports_chat_template=True,
        image_token="",
        supports_chat_api=True,
        default_models=[
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        ],
    ),
    "glm4v": FamilyConfig(
        auto_class="GLM4VForConditionalGeneration",
        supports_chat_template=True,
        image_token="",
        supports_chat_api=True,
        default_models=[
            "zai-org/GLM-4.5V",
            "zai-org/GLM-4.6V-Flash",
        ],
    ),
    # ---- Legacy models (no inference provider — local only) ----
    "llava": FamilyConfig(
        auto_class="LlavaForConditionalGeneration",
        supports_chat_template=True,
        image_token="<image>",
        supports_chat_api=True,
        default_models=[
            "llava-hf/llava-interleave-qwen-0.5b-hf",
            "llava-hf/llava-1.5-7b-hf",
        ],
    ),
    "llava_next": FamilyConfig(
        auto_class="LlavaNextForConditionalGeneration",
        supports_chat_template=True,
        image_token="<image>",
        supports_chat_api=True,
        default_models=[
            "llava-hf/llava-v1.6-mistral-7b-hf",
        ],
    ),
    "qwen2_vl": FamilyConfig(
        auto_class="Qwen2VLForConditionalGeneration",
        supports_chat_template=True,
        image_token="<|image_pad|>",
        supports_chat_api=True,
        default_models=[
            "Qwen/Qwen2-VL-7B-Instruct",
        ],
    ),
    "paligemma": FamilyConfig(
        auto_class="PaliGemmaForConditionalGeneration",
        supports_chat_template=False,
        image_token="",
        supports_chat_api=False,
        default_models=[
            "google/paligemma-3b-mix-224",
        ],
    ),
    "instructblip": FamilyConfig(
        auto_class="InstructBlipForConditionalGeneration",
        supports_chat_template=False,
        image_token="",
        supports_chat_api=False,
        default_models=[
            "Salesforce/instructblip-vicuna-7b",
        ],
    ),
}


def detect_model_family(model_id: str) -> str:
    """Infer the model family from a HuggingFace model id or local path.

    Uses name heuristics first, then falls back to reading the HF config.
    """
    # Use full model_id (lowercased) for multi-segment matching
    full_lower = model_id.lower()
    name = os.path.basename(model_id.rstrip("/")).lower()

    # Name heuristics — check newer/more-specific families first
    # Qwen3-VL before Qwen2.5-VL before Qwen2-VL
    if "qwen3-vl" in full_lower or "qwen3_vl" in full_lower:
        return "qwen3_vl"
    if "qwen2.5-vl" in full_lower or "qwen2_5_vl" in full_lower:
        return "qwen2_5_vl"
    if "qwen2-vl" in full_lower or "qwen2_vl" in full_lower:
        return "qwen2_vl"
    # Gemma 3 (including gemma-3n)
    if "gemma-3" in name or "gemma3" in name:
        return "gemma3"
    # Llama 3.2 Vision (Mllama) — must check before Llama 4
    if ("llama-3" in full_lower or "llama3" in full_lower) and (
        "vision" in full_lower or "11b" in name or "90b" in name
    ):
        return "mllama"
    # Llama 4
    if "llama-4" in name or "llama4" in name:
        return "llama4"
    # GLM-4.5V / GLM-4.6V
    if "glm-4" in name and ("v" in name or "vision" in name):
        return "glm4v"
    if "glm4" in name:
        return "glm4v"
    # PaliGemma (before gemma generic — but "gemma-3" already matched above)
    if "paligemma" in name:
        return "paligemma"
    if "instructblip" in name:
        return "instructblip"
    if "llava" in name:
        if "v1.6" in name or "next" in name or "1.6" in name:
            return "llava_next"
        return "llava"

    # Fall back to HF config inspection
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        arch = cfg.architectures[0] if getattr(cfg, "architectures", None) else ""
        arch_lower = arch.lower()
        if "qwen3vl" in arch_lower:
            return "qwen3_vl"
        if "qwen2_5_vl" in arch_lower or "qwen2.5vl" in arch_lower:
            return "qwen2_5_vl"
        if "qwen2vl" in arch_lower:
            return "qwen2_vl"
        if "gemma3" in arch_lower:
            return "gemma3"
        if "mllama" in arch_lower:
            return "mllama"
        if "llama4" in arch_lower:
            return "llama4"
        if "glm4v" in arch_lower or "glm4" in arch_lower:
            return "glm4v"
        if "paligemma" in arch_lower:
            return "paligemma"
        if "instructblip" in arch_lower:
            return "instructblip"
        if "llavanext" in arch_lower:
            return "llava_next"
        if "llava" in arch_lower:
            return "llava"
    except Exception:
        pass

    raise ValueError(
        f"Cannot detect model family for '{model_id}'. "
        f"Pass family= explicitly. Supported: {list(MODEL_FAMILY_CONFIGS.keys())}"
    )


def list_supported_families() -> List[str]:
    """Return all supported model family names."""
    return list(MODEL_FAMILY_CONFIGS.keys())


def list_default_models(family: Optional[str] = None) -> Dict[str, List[str]]:
    """Return default model ids, optionally filtered by family."""
    if family:
        cfg = MODEL_FAMILY_CONFIGS.get(family)
        if cfg is None:
            raise ValueError(f"Unknown family '{family}'. Supported: {list_supported_families()}")
        return {family: cfg.default_models}
    return {k: v.default_models for k, v in MODEL_FAMILY_CONFIGS.items()}


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64-encoded data URI."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _load_image(sample: CanonicalSample, image_root: str = "") -> Image.Image:
    """Load a PIL Image from a CanonicalSample."""
    pil_image = sample.metadata.get("_pil_image") if sample.metadata else None
    if pil_image is not None:
        return pil_image.convert("RGB")
    img_path = (
        os.path.join(image_root, sample.image_path)
        if image_root
        else sample.image_path
    )
    return Image.open(img_path).convert("RGB")


# ---------------------------------------------------------------------------
# HFMethod — API-based inference (primary mode)
# ---------------------------------------------------------------------------

class HFMethod(Method):
    """HuggingFace VLM inference wrapper using HF Inference API.

    By default, calls the HF Inference API (serverless). No local model
    download needed.

    Parameters
    ----------
    model_id:
        HuggingFace model id (e.g. ``"llava-hf/llava-1.5-7b-hf"``).
    family:
        Model family key. Auto-detected if not provided.
    hf_token:
        HuggingFace API token. Uses ``HF_TOKEN`` env var if not set.
    max_new_tokens:
        Default max tokens for generation.
    temperature:
        Default temperature for generation.
    """

    def __init__(
        self,
        model_id: str,
        family: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.family = family or detect_model_family(model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._family_cfg = MODEL_FAMILY_CONFIGS[self.family]
        self._client = None

    @property
    def client(self):
        """Lazy-init the HF InferenceClient.

        HF Inference API requires a token for inference calls.
        Token is optional for ``test_connection()`` (which only queries model info).
        """
        if self._client is None:
            from huggingface_hub import InferenceClient
            token = self._hf_token
            if not token:
                raise ValueError(
                    "推理需要 HuggingFace API token。\n"
                    "获取方式：https://huggingface.co/settings/tokens\n"
                    "设置方式：export HF_TOKEN=hf_xxx 或在 demo 界面中输入"
                )
            self._client = InferenceClient(
                model=self.model_id,
                token=token,
            )
        return self._client

    def update_token(self, token: str) -> None:
        """Update the API token and reset the client so it picks up the new token."""
        self._hf_token = token or None
        self._client = None  # force re-creation on next access

    # ------------------------------------------------------------------
    # Method interface
    # ------------------------------------------------------------------

    def inference(
        self,
        question: str,
        image: Any,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run inference via HF Inference API.

        Parameters
        ----------
        question:
            The question to ask about the image.
        image:
            PIL Image, file path (str), or bytes.
        max_new_tokens:
            Override default max tokens.
        temperature:
            Override default temperature.

        Returns
        -------
        str
            Model's text response.
        """
        _max_tokens = max_new_tokens or self.max_new_tokens
        _temperature = temperature if temperature is not None else self.temperature

        # Convert image to PIL if needed
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if self._family_cfg.supports_chat_api:
            return self._inference_chat(question, pil_image, _max_tokens, _temperature)
        else:
            return self._inference_vqa(question, pil_image)

    def inference_with_metadata(
        self,
        question: str,
        image: Any,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Like inference(), but returns metadata alongside the answer.

        Returns
        -------
        dict
            ``{"answer": str, "usage": {"prompt_tokens": int, "completion_tokens": int} | None}``
        """
        _max_tokens = max_new_tokens or self.max_new_tokens
        _temperature = temperature if temperature is not None else self.temperature

        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if self._family_cfg.supports_chat_api:
            return self._inference_chat_with_metadata(
                question, pil_image, _max_tokens, _temperature,
            )
        else:
            answer = self._inference_vqa(question, pil_image)
            return {"answer": answer, "usage": None}

    def _inference_chat(
        self, question: str, image: Image.Image,
        max_new_tokens: int, temperature: float,
    ) -> str:
        """Use chat_completion API for chat-capable models."""
        result = self._inference_chat_with_metadata(
            question, image, max_new_tokens, temperature,
        )
        return result["answer"]

    def _inference_chat_with_metadata(
        self, question: str, image: Image.Image,
        max_new_tokens: int, temperature: float,
    ) -> Dict[str, Any]:
        """Chat completion that returns answer + usage + request metadata."""
        image_uri = _pil_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content.strip()
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            }

        # Build sanitized request payload (truncate base64 image)
        img_len = len(image_uri)
        request_payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"[base64 image, {img_len} chars]"}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        return {"answer": answer, "usage": usage, "request": request_payload}

    def _inference_vqa(self, question: str, image: Image.Image) -> str:
        """Use visual_question_answering API for non-chat models."""
        results = self.client.visual_question_answering(
            image=image,
            question=question,
        )
        if results:
            return results[0].answer if hasattr(results[0], "answer") else str(results[0])
        return ""

    def inference_sample(
        self,
        sample: CanonicalSample,
        image_root: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run inference on a CanonicalSample."""
        image = _load_image(sample, image_root)
        question = ""
        for turn in sample.turns:
            if turn.role == "human":
                question = turn.content
                break
        if not question:
            question = "Describe this image."
        return self.inference(question, image, max_new_tokens, temperature)

    def test_connection(self) -> Dict[str, Any]:
        """Test if the model is accessible via HF API.

        Returns a dict with status info.
        """
        try:
            from huggingface_hub import model_info
            info = model_info(self.model_id, token=self._hf_token)
            return {
                "status": "ok",
                "model_id": self.model_id,
                "family": self.family,
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "private": getattr(info, "private", False),
                "gated": getattr(info, "gated", False),
                "library_name": getattr(info, "library_name", None),
            }
        except Exception as e:
            return {
                "status": "error",
                "model_id": self.model_id,
                "family": self.family,
                "error": str(e),
            }

    # Compatibility — prepare_input / generate for local mode
    def prepare_input(
        self,
        sample: CanonicalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Prepare input for inference (API mode — just packages data)."""
        image = _load_image(sample, image_root)
        question = ""
        for turn in sample.turns:
            if turn.role == "human":
                question = turn.content
                break
        return {"question": question or "Describe this image.", "image": image}

    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Run inference using prepared input."""
        return self.inference(
            prepared["question"], prepared["image"],
            max_new_tokens, temperature,
        )
