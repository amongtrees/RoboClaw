"""World Model remote API client.

Calls a World Model server (e.g. NVIDIA Cosmos Predict-2 or Qwen-RobotWorld)
to predict future visual observations and robot states given a proposed
action.  Used by ``reflect_node`` for predictive safety lookahead.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from roboclaw.core.config import ModelClientConfig
from roboclaw.core.types import ModelType
from roboclaw.models.clients import (
    AbstractModelClient,
    ModelInferenceRequest,
    ModelInferenceResponse,
)

logger = logging.getLogger(__name__)


class WorldModelClient(AbstractModelClient):
    """Remote client for a World Model server.

    Two deployment modes (transparent to the caller):

    1. **Self-hosted** — a local FastAPI server wrapping Cosmos Predict-2
       or Qwen-RobotWorld weights.
    2. **Proxy mode** — forward requests to NVIDIA's hosted Cosmos API at
       ``api.build.nvidia.com``.  Set ``base_url`` to the provider URL
       and supply ``api_key``.

    Input:  current RGB frame(s) + robot state + proposed action description.
    Output: predicted future frame(s) + predicted robot state + uncertainty.
    """

    def __init__(self, config: ModelClientConfig) -> None:
        self._config = config
        self._api_key: str = self._resolve_api_key(config)
        self._client: httpx.AsyncClient | None = None
        self._available: bool = False

    # ------------------------------------------------------------------
    # AbstractModelClient interface
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return ModelType.WORLD_MODEL

    @property
    def is_available(self) -> bool:
        return self._available

    async def health_check(self) -> bool:
        client = await self._ensure_client()
        try:
            resp = await client.get("/health", timeout=httpx.Timeout(5.0))
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def infer(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        client = await self._ensure_client()
        try:
            response = await client.post(
                self._config.api_path,
                json=request.model_dump(mode="json"),
                timeout=httpx.Timeout(self._config.timeout_sec),
            )
            response.raise_for_status()
            data = response.json()
            return ModelInferenceResponse(**data)
        except httpx.TimeoutException:
            logger.error("World Model inference timed out after %.1fs", self._config.timeout_sec)
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "wm_timeout", "detail": f"Timeout after {self._config.timeout_sec}s"},
            )
        except httpx.HTTPStatusError as e:
            logger.error("World Model HTTP %d: %s", e.response.status_code, e.response.text[:200])
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "wm_http_error", "detail": str(e)},
            )
        except Exception as e:
            logger.error("World Model inference failed: %s", e)
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "wm_api_error", "detail": str(e)},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key(config: ModelClientConfig) -> str:
        """Resolve ``${ENV_VAR}`` placeholders in the api_key field."""
        key = config.api_key
        if key.startswith("${") and key.endswith("}"):
            env_var = key[2:-1]
            return os.environ.get(env_var, "")
        return key

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = dict(self._config.headers) if self._config.headers else {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(base_url=self._config.base_url, headers=headers)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
