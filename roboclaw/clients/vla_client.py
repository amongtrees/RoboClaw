"""VLA (Vision-Language-Action) remote API client.

Calls a VLA model microservice (e.g. GR00T-N1-2B) to convert RGB images
and a manipulation instruction into joint actions, end-effector poses,
and gripper commands.
"""

from __future__ import annotations

import logging
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


class VLAClient(AbstractModelClient):
    """Remote client for a VLA model server (e.g. GR00T-N1-2B, pi0).

    Sends ``POST /v1/act`` with RGB images + proprioceptive state + instruction.
    Receives joint actions, end-effector poses, and a gripper command.

    GR00T-N1 uses a dual-system architecture:
    - System 2 (VLM): reasons about *what* to do
    - System 1 (DiT / flow-matching): generates *how* to do it at ~16 Hz

    When unavailable the ``ModelRouter`` falls back to ``ManipulationExecutor``.
    """

    def __init__(self, config: ModelClientConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._available: bool = False

    # ------------------------------------------------------------------
    # AbstractModelClient interface
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return ModelType.VLA

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

    async def warm_up(self) -> bool:
        """Send a dummy manipulation request to warm GPU kernels."""
        if not await self.health_check():
            return False
        try:
            req = ModelInferenceRequest(
                model_type=ModelType.VLA,
                sub_task_id="warmup",
                skill_type="whole_body_grasp",
                parameters={"object": "warmup", "arm": "right"},
                goal="warmup",
                instruction="Grasp the warmup object",
            )
            resp = await self.infer(req)
            return resp.is_success
        except Exception:
            return False

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
            logger.error("VLA inference timed out after %.1fs", self._config.timeout_sec)
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "vla_timeout", "detail": f"Timeout after {self._config.timeout_sec}s"},
            )
        except httpx.HTTPStatusError as e:
            logger.error("VLA inference HTTP %d: %s", e.response.status_code, e.response.text[:200])
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "vla_http_error", "detail": str(e)},
            )
        except Exception as e:
            logger.error("VLA inference failed: %s", e)
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "vla_api_error", "detail": str(e)},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = dict(self._config.headers) if self._config.headers else {"Content-Type": "application/json"}
            self._client = httpx.AsyncClient(base_url=self._config.base_url, headers=headers)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
