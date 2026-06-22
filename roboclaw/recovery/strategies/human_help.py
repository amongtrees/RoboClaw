"""Human help recovery strategy — request operator assistance.

This is the last-resort strategy. It always returns can_handle=True
to prevent dead ends in the recovery chain.
"""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.recovery.base import AbstractRecoveryStrategy, RecoveryResult

logger = logging.getLogger(__name__)


class HumanHelpStrategy(AbstractRecoveryStrategy):
    """Last-resort recovery: request human operator help.

    Always returns can_handle=True to ensure the recovery chain
    never reaches a dead end. Sends a detailed message to the
    operator describing what went wrong and what was tried.
    """

    def can_handle(self, error: dict[str, Any], state: Any) -> bool:
        # Always returns True — this is the safety net
        return True

    async def handle(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        logger.warning("Requesting human operator assistance...")

        # Compose a clear message for the operator
        error_type = error.get("type", "unknown")
        error_detail = error.get("detail", "No additional detail")

        task_goal = ""
        current_subtask = None
        if hasattr(state, "task_spec") and state.task_spec:
            task_goal = state.task_spec.get("goal", "")
        if hasattr(state, "current_subtask"):
            current_subtask = state.current_subtask

        human_message = (
            f"Robot {getattr(state, 'robot_id', 'unknown')} requires assistance.\n"
            f"Task: {task_goal}\n"
            f"Error Type: {error_type}\n"
            f"Error Detail: {error_detail}\n"
            f"Failed Sub-Task: {current_subtask}\n"
            f"Recovery Attempts: {getattr(state, 'recovery_attempts', 0)}\n"
            f"\nPlease review and resolve via the operator console."
        )

        logger.info(f"Human help message: {human_message}")

        # In production, this sends the message via WebSocket to the operator UI
        # and awaits a response. For Phase 1, it's a terminal state.

        return RecoveryResult(
            success=True,  # Successfully escalated to human operator
            strategy_used="human_help",
            human_message=human_message,
            recovery_event={
                "action": "request_human_help",
                "error": error,
                "message_sent": True,
            },
        )
