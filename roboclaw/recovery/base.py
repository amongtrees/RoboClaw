"""Abstract base for recovery strategies."""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field


class RecoveryResult(BaseModel):
    """Result of a recovery attempt."""

    success: bool = False
    strategy_used: str = ""
    modified_plan: dict[str, Any] | None = None
    modified_subtask: dict[str, Any] | None = None
    recovery_event: dict[str, Any] = Field(default_factory=dict)
    human_message: str | None = None


class AbstractRecoveryStrategy(abc.ABC):
    """Abstract interface for a failure recovery strategy."""

    @abc.abstractmethod
    def can_handle(self, error: dict[str, Any], state: Any) -> bool:
        """Check if this strategy can handle the given error.

        Args:
            error: Error dict with 'type' and 'detail' fields.
            state: Current AgentGraphState.

        Returns:
            True if this strategy should be tried.
        """
        ...

    @abc.abstractmethod
    async def handle(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        """Attempt recovery.

        Args:
            error: The error to recover from.
            state: Current AgentGraphState.

        Returns:
            RecoveryResult indicating success/failure and any modifications.
        """
        ...
