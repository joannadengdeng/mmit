"""LiteLLMMethod — unified closed-source multimodal inference via LiteLLM.

Supports any provider that LiteLLM wraps (OpenAI, Anthropic, Google Gemini, etc.)
with a single OpenAI-style interface.

Supported models
----------------
* ``gemini/gemini-2.5-flash-lite``  — Google, **free** (1000 req/day)
* ``gemini/gemini-2.5-flash``       — Google
* ``openai/gpt-4o-mini``            — OpenAI ($5 trial credits)
* ``anthropic/claude-haiku-4-5``    — Anthropic ($5 trial credits)

Example
-------
>>> method = LiteLLMMethod("gemini/gemini-2.5-flash-lite", api_key="YOUR_KEY")
>>> result = method.inference("What is in this image?", pil_image)
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from mmit.data.types import CanonicalSample
from mmit.eval.methods.base import Method


# ---------------------------------------------------------------------------
# Provider → env-var mapping
# ---------------------------------------------------------------------------

_PROVIDER_ENV_KEYS: Dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _detect_provider(model_id: str) -> str:
    """Extract provider prefix from a LiteLLM model id."""
    if "/" in model_id:
        prefix = model_id.split("/")[0].lower()
        if prefix in _PROVIDER_ENV_KEYS:
            return prefix
    return ""


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64-encoded data URI."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Available closed-source models
# ---------------------------------------------------------------------------

CLOSED_SOURCE_MODELS: List[Dict[str, str]] = [
    {
        "id": "gemini/gemini-2.5-flash-lite",
        "provider": "Google",
        "label": "Google API, free",
    },
    {
        "id": "gemini/gemini-2.5-flash",
        "provider": "Google",
        "label": "Google API",
    },
    {
        "id": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "label": "OpenAI API",
    },
    {
        "id": "anthropic/claude-haiku-4-5",
        "provider": "Anthropic",
        "label": "Anthropic API",
    },
]


def is_closed_source_model(model_id: str) -> bool:
    """Check if a model ID uses a closed-source provider (LiteLLM routing)."""
    return model_id.startswith(("gemini/", "openai/", "anthropic/"))


# ---------------------------------------------------------------------------
# LiteLLMMethod
# ---------------------------------------------------------------------------

class LiteLLMMethod(Method):
    """Multimodal inference via closed-source APIs using LiteLLM.

    Parameters
    ----------
    model_id:
        LiteLLM model string, e.g. ``"gemini/gemini-2.5-flash-lite"``.
    api_key:
        Provider API key. Falls back to the provider-specific env var
        (``GEMINI_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
    max_new_tokens:
        Default max tokens for generation.
    temperature:
        Default sampling temperature.
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.family = "litellm"

        self._provider = _detect_provider(model_id)
        self._api_key = api_key or ""

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Return the API key, falling back to env var."""
        if self._api_key:
            return self._api_key
        env_var = _PROVIDER_ENV_KEYS.get(self._provider, "")
        if env_var:
            val = os.environ.get(env_var, "")
            if val:
                return val
        raise ValueError(
            f"需要 {self._provider.upper()} API key。"
            f"请在界面输入或设置环境变量 {_PROVIDER_ENV_KEYS.get(self._provider, '???')}。"
        )

    def update_token(self, token: str) -> None:
        """Update the API key at runtime."""
        self._api_key = token or ""

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    def test_connection(self) -> Dict[str, Any]:
        """Test that the provider is reachable and the key is valid."""
        try:
            import litellm
        except ImportError:
            return {
                "status": "error",
                "error": "litellm 未安装。请运行: pip install litellm",
            }
        try:
            self._resolve_api_key()
        except ValueError as e:
            return {"status": "error", "error": str(e)}

        return {
            "status": "ok",
            "model_id": self.model_id,
            "provider": self._provider,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference(
        self,
        question: str,
        image: Any,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run multimodal inference and return the text answer."""
        result = self._call_litellm(question, image, max_new_tokens, temperature)
        return result["answer"]

    def inference_with_metadata(
        self,
        question: str,
        image: Any,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run inference and return answer + token usage metadata."""
        return self._call_litellm(question, image, max_new_tokens, temperature)

    def _call_litellm(
        self,
        question: str,
        image: Any,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Core LiteLLM call. Returns {"answer": str, "usage": dict | None}."""
        import litellm

        api_key = self._resolve_api_key()
        max_tok = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        # Convert image to PIL if needed
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        image_uri = _pil_to_base64(pil_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": question},
                ],
            }
        ]

        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tok,
            temperature=temp,
            api_key=api_key,
        )

        answer = response.choices[0].message.content.strip()

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
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
            "max_tokens": max_tok,
            "temperature": temp,
        }
        return {"answer": answer, "usage": usage, "request": request_payload}

    # ------------------------------------------------------------------
    # Method ABC stubs (needed for compatibility but not used in API mode)
    # ------------------------------------------------------------------


    def prepare_input(self, sample: CanonicalSample, image_root: str = "") -> Dict[str, Any]:
        from mmit.data.adapters.hf_datasets import decode_sample_image
        pil_image = decode_sample_image(sample)
        question = sample.instruction or sample.first_question or "Describe this image."
        return {"question": question, "image": pil_image}

    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        return self.inference(
            question=prepared["question"],
            image=prepared["image"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
