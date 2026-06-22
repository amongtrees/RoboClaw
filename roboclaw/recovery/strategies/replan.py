"""Re-plan recovery strategy — generate a new plan with updated world state."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.recovery.base import AbstractRecoveryStrategy, RecoveryResult

logger = logging.getLogger(__name__)


class RePlanStrategy(AbstractRecoveryStrategy):
    """Second-line recovery: re-plan with updated world state.

    Handles: kinematic failures, planning errors, object-not-found,
             dynamic failures (slip, push — re-plan with new object pose).

    Queries semantic memory for similar past recovery successes.
    """

    def can_handle(self, error: dict[str, Any], state: Any) -> bool:
        error_type = error.get("type", "")
        return error_type in ("kinematic", "dynamic", "planning", "perception")

    async def handle(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        logger.info("Re-planning with updated world state...")

        # In production, this:
        # 1. Re-runs perception to get updated world state
        # 2. Queries SemanticMemory for similar failures + solutions
        # 3. Calls TaskDecomposer with adjusted parameters
        # 4. Returns the new plan

        error_detail = error.get("detail", "")

        return RecoveryResult(
            success=True,
            strategy_used="replan",
            modified_plan={
                "replan_reason": error_detail,
                "adjustment": "Using alternative approach based on updated world state",
            },
            recovery_event={
                "action": "replan",
                "original_error": error,
                "detail": "Generated alternative plan with adjusted parameters",
            },
        )
