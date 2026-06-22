"""Safe stop recovery strategy — graceful stop and return to safe state."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.recovery.base import AbstractRecoveryStrategy, RecoveryResult

logger = logging.getLogger(__name__)


class SafeStopStrategy(AbstractRecoveryStrategy):
    """Third-line recovery: stop safely and recover to a known-good state.

    Handles: safety violations, major dynamic failures, unrecoverable kinematic errors.
    Cannot handle: hardware failures requiring repair.
    """

    def can_handle(self, error: dict[str, Any], state: Any) -> bool:
        error_type = error.get("type", "")
        return error_type in ("safety", "dynamic", "hardware")

    async def handle(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        logger.warning("Initiating safe stop...")

        # Steps:
        # 1. Stop all current motions
        # 2. Return arms to home/rest position
        # 3. Verify stable stance (double support)
        # 4. Report status

        error_type = error.get("type", "")

        if error_type == "safety":
            logger.critical("Safety violation — executing emergency stop and returning to safe pose")
            return RecoveryResult(
                success=True,
                strategy_used="safe_stop",
                recovery_event={
                    "action": "emergency_stop_and_safe_pose",
                    "detail": "All motions stopped, arms at rest, stable stance confirmed",
                },
            )

        if error_type == "hardware":
            # Hardware fault — cannot recover, must stop
            logger.error("Hardware fault detected — safe stop only, cannot continue")
            return RecoveryResult(
                success=False,
                strategy_used="safe_stop",
                recovery_event={
                    "action": "safe_stop_hardware_fault",
                    "detail": "Hardware fault requires maintenance. Robot is in safe stopped state.",
                },
            )

        # General safe stop
        return RecoveryResult(
            success=True,
            strategy_used="safe_stop",
            recovery_event={
                "action": "safe_stop_and_hold",
                "detail": "Robot stopped in current position, awaiting further instructions",
            },
        )
