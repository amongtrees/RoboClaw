"""VLN (Vision-Language Navigation) remote API client.

Calls a VLN model microservice (e.g. Qwen-RobotNav) to convert RGB images
and a navigation goal into waypoints / trajectories.
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


class VLNClient(AbstractModelClient):
    """Remote client for a VLN model server (e.g. Qwen-RobotNav).

    Sends ``POST /v1/navigate`` with RGB images + goal instruction.
    Receives waypoints and a trajectory.

    When the remote server is unreachable the client automatically marks
    itself unavailable; the ``ModelRouter`` will then fall back to the
    in-process ``LocomotionExecutor`` stub.
    """

    def __init__(self, config: ModelClientConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._available: bool = False
        self._waypoint_cache: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # AbstractModelClient interface
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return ModelType.VLN

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
        """Send a dummy navigation request to warm GPU kernels."""
        if not await self.health_check():
            return False
        try:
            req = ModelInferenceRequest(
                model_type=ModelType.VLN,
                sub_task_id="warmup",
                skill_type="navigate_to",
                parameters={"room": "warmup"},
                goal="warmup",
                instruction="Navigate to warmup",
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
            result = ModelInferenceResponse(**data)
            if result.waypoints:
                self._waypoint_cache = result.waypoints
            return result
        except httpx.TimeoutException:
            logger.error("VLN inference timed out after %.1fs", self._config.timeout_sec)
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "vln_timeout", "detail": f"Timeout after {self._config.timeout_sec}s"},
            )
        except httpx.HTTPStatusError as e:
            logger.error("VLN inference HTTP %d: %s", e.response.status_code, e.response.text[:200])
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "vln_http_error", "detail": str(e)},
            )
        except Exception as e:
            logger.error("VLN inference failed: %s", e)
            return ModelInferenceResponse(
                request_id=request.request_id,
                status="failure",
                error={"type": "vln_api_error", "detail": str(e)},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=self._config.headers or {"Content-Type": "application/json"},
            )
        return self._client

    def pop_waypoints(self) -> list[dict[str, float]]:
        """Return cached waypoints and clear the cache."""
        pts = list(self._waypoint_cache)
        self._waypoint_cache.clear()
        return pts

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
