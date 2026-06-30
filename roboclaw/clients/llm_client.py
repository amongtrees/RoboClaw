"""OpenAI-compatible LLM client for task decomposition and planning.

Supports any OpenAI-compatible API endpoint (vLLM, Ollama, LiteLLM,
GPT-4o, etc.).  Implements the ``ainvoke(messages)`` interface expected
by ``TaskDecomposer``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """Async client for any OpenAI-compatible chat completions API.

    Parameters
    ----------
    base_url:
        API base URL, e.g. ``"http://localhost:8080/v1"``.
    api_key:
        API key (or ``"not-needed"`` for local servers).
    model:
        Model name to use, e.g. ``"gpt-4o"``, ``"Qwen2.5-7B"``.
    temperature:
        Sampling temperature (0.0 = deterministic).
    max_tokens:
        Maximum output tokens.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        # Lazy-import httpx to avoid hard dependency
        self._client: Any = None

    # ------------------------------------------------------------------
    # LangChain-compatible interface
    # ------------------------------------------------------------------

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Send chat completion request.

        Parameters
        ----------
        messages:
            List of ``langchain_core.messages.SystemMessage``,
            ``HumanMessage``, or plain dicts with ``role``/``content``.

        Returns
        -------
            An object with a ``.content`` string attribute (mimics
            LangChain's ``AIMessage``).
        """
        payload_messages = []
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                # LangChain message object
                role = "system" if msg.type == "system" else "user"
                payload_messages.append({
                    "role": role,
                    "content": _extract_content(msg.content),
                })
            elif isinstance(msg, dict):
                payload_messages.append(msg)
            else:
                payload_messages.append({"role": "user", "content": str(msg)})

        payload = {
            "model": self._model,
            "messages": payload_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self._timeout)

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                url, headers=headers, json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            raise

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        logger.debug(
            "LLM response: %d tokens (prompt=%d, completion=%d), model=%s",
            usage.get("total_tokens", 0),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            data.get("model", self._model),
        )

        return _LLMResponse(content)


def _extract_content(content: Any) -> str:
    """Handle LangChain content which can be str or list[dict]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Content blocks: [{"type": "text", "text": "..."}, ...]
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


class _LLMResponse:
    """Minimal AIMessage-compatible response wrapper."""

    def __init__(self, content: str) -> None:
        self.content = content

    def __repr__(self) -> str:
        return f"_LLMResponse({self.content[:80]}...)"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_llm_client_from_config(llm_config: Any) -> OpenAICompatibleClient | None:
    """Create an ``OpenAICompatibleClient`` from a config object.

    Returns ``None`` if no ``api_base`` is configured (template fallback
    will be used by the TaskDecomposer).
    """
    api_base = getattr(llm_config, "api_base", None)
    if not api_base:
        # Also check dict-style config
        if isinstance(llm_config, dict):
            api_base = llm_config.get("api_base", "")
        if not api_base:
            logger.info("No LLM api_base configured — template planning only")
            return None

    if isinstance(llm_config, dict):
        return OpenAICompatibleClient(
            base_url=str(llm_config.get("api_base", "http://localhost:8080/v1")),
            api_key=str(llm_config.get("api_key", "")),
            model=str(llm_config.get("planning", {}).get("model", llm_config.get("model", "gpt-4o"))),
            temperature=float(llm_config.get("planning", {}).get("temperature", 0.1)),
            max_tokens=int(llm_config.get("planning", {}).get("max_tokens", 8192)),
        )

    return OpenAICompatibleClient(
        base_url=api_base,
        api_key=getattr(llm_config, "api_key", ""),
        model=getattr(
            getattr(llm_config, "planning", None), "model",
            getattr(llm_config, "model", "gpt-4o"),
        ),
        temperature=getattr(
            getattr(llm_config, "planning", None), "temperature", 0.1,
        ),
        max_tokens=getattr(
            getattr(llm_config, "planning", None), "max_tokens", 8192,
        ),
    )
