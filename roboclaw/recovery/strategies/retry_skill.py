"""Retry skill recovery strategy — retry with parameter adjustment."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.recovery.base import AbstractRecoveryStrategy, RecoveryResult

logger = logging.getLogger(__name__)


class RetrySkillStrategy(AbstractRecoveryStrategy):
    """Retry the failed skill with adjusted parameters.

    Handles: transient failures where the skill itself is valid but
             execution conditions were suboptimal (e.g., slightly off approach angle).
    """

    def can_handle(self, error: dict[str, Any], state: Any) -> bool:
        error_type = error.get("type", "")
        return error_type in ("kinematic", "dynamic", "perception")

    async def handle(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        logger.info("Retrying skill with parameter adjustment...")

        current_subtask = None
        if hasattr(state, "current_subtask"):
            current_subtask = state.current_subtask

        if current_subtask is None:
            return RecoveryResult(
                success=False,
                strategy_used="retry_skill",
                recovery_event={"detail": "No current sub-task to retry"},
            )

        # Adjust parameters based on error type
        adjusted = dict(current_subtask)
        params = adjusted.get("parameters", {})

        error_type = error.get("type", "")
        if error_type == "kinematic":
            # Try a different approach angle or arm
            if params.get("arm") == "right":
                params["arm"] = "left"
            params["approach_offset_m"] = params.get("approach_offset_m", 0.1) + 0.05

        elif error_type == "dynamic":
            # Slow down the execution
            params["speed_factor"] = params.get("speed_factor", 1.0) * 0.7

        elif error_type == "perception":
            # Try with different camera settings or wider search area
            params["search_radius_m"] = params.get("search_radius_m", 0.5) * 1.5

        adjusted["parameters"] = params

        return RecoveryResult(
            success=True,
            strategy_used="retry_skill",
            modified_subtask=adjusted,
            recovery_event={
                "action": "retry_with_adjustment",
                "original_params": current_subtask.get("parameters", {}),
                "adjusted_params": params,
            },
        )
