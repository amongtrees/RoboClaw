"""Abstract base for action executors and the ActionResult model."""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActionResultStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    INCOMPLETE = "incomplete"
    INTERRUPTED = "interrupted"


class ActionResult(BaseModel):
    """Result of executing a sub-task."""

    sub_task_id: str
    status: ActionResultStatus = ActionResultStatus.SUCCESS
    duration_sec: float = 0.0
    actual_outcome: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    artifacts: list[str] = Field(default_factory=list)  # artifact_ids


class AbstractActionExecutor(abc.ABC):
    """Abstract interface for action executors (locomotion, manipulation, etc.)."""

    @abc.abstractmethod
    async def execute(self, sub_task: dict[str, Any], world_state: dict[str, Any]) -> ActionResult:
        """Execute a sub-task and return the result.

        Args:
            sub_task: The SubTask dict with skill_type, parameters, etc.
            world_state: Current world state from perception/memory.

        Returns:
            ActionResult with status, outcome, and any error info.
        """
        ...

    @abc.abstractmethod
    async def cancel(self) -> None:
        """Cancel the currently executing action (graceful stop)."""
        ...

    @abc.abstractmethod
    async def emergency_stop(self) -> None:
        """Immediate emergency stop (hard stop)."""
        ...
