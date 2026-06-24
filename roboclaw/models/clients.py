"""Abstract model client base and standard inference request/response schemas.

This module defines the contract that all VLN, VLA, and World Model clients
must satisfy.  Every remote model is called via the same Pydantic envelopes
so that the ModelRouter and the orchestration layer can treat them uniformly.
"""

from __future__ import annotations

import abc
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _new_request_id() -> str:
    return f"req_{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Standard inference envelopes
# ---------------------------------------------------------------------------


class ModelInferenceRequest(BaseModel):
    """Payload sent to any model microservice (VLN / VLA / World Model).

    Images are passed as **base64-encoded strings** (JPEG recommended for
    bandwidth efficiency).  The ``robot_state`` dict carries the robot's
    current proprioceptive state (joint positions, end-effector poses, ZMP
    etc.) serialised as JSON-compatible primitives.
    """

    request_id: str = Field(default_factory=_new_request_id)
    model_type: str = ""  # ModelType enum value
    sub_task_id: str = ""
    skill_type: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    # ---- Vision inputs (base64-encoded JPEG / PNG) ----
    rgb_images: list[str] = Field(default_factory=list)
    depth_images: list[str] = Field(default_factory=list)

    # ---- Language & context ----
    goal: str = ""
    instruction: str = ""
    robot_state: dict[str, Any] = Field(default_factory=dict)
    world_state: dict[str, Any] = Field(default_factory=dict)

    # ---- Misc ----
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInferenceResponse(BaseModel):
    """Standardised response returned by every model microservice.

    Different model types populate different subsets of fields:

    * **VLN** → ``waypoints``, ``trajectory``
    * **VLA** → ``joint_actions``, ``ee_poses``, ``gripper_command``,
      ``action_tokens``
    * **World Model** → ``predicted_frames``, ``predicted_state``,
      ``uncertainty``
    """

    request_id: str
    status: str = "success"  # success | failure | partial

    # ---- VLN outputs ----
    waypoints: list[dict[str, float]] = Field(default_factory=list)
    trajectory: list[dict[str, float]] = Field(default_factory=list)

    # ---- VLA outputs ----
    joint_actions: dict[str, list[float]] = Field(default_factory=dict)
    ee_poses: list[dict[str, float]] = Field(default_factory=list)
    gripper_command: str | None = None  # "open" | "close" | "hold"
    action_tokens: list[int] = Field(default_factory=list)

    # ---- World Model outputs ----
    predicted_frames: list[str] = Field(default_factory=list)  # base64 images
    predicted_state: dict[str, Any] = Field(default_factory=dict)
    uncertainty: float = 0.0

    # ---- Common ----
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "success"


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class AbstractModelClient(abc.ABC):
    """Contract for every remote model client.

    Sub-classes wrap an ``httpx.AsyncClient`` pointed at a model-serving
    FastAPI microservice (or a hosted API like NVIDIA Cosmos).

    Every client is *optional* — when a model is not available the system
    falls back to the in-process simulated executor transparently.
    """

    @abc.abstractmethod
    async def infer(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        """Send an inference request and return the structured response."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the remote model server is reachable and healthy."""

    async def warm_up(self) -> bool:
        """Optional: send a dummy request to warm GPU kernels / caches."""
        return True

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """The ModelType enum value this client handles."""

    @property
    def is_available(self) -> bool:
        """Synchronous convenience — returns the cached availability flag."""
        return getattr(self, "_available", False)
