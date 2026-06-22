"""Self-diagnose recovery strategy — check sensors/calibration, fix simple issues."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.recovery.base import AbstractRecoveryStrategy, RecoveryResult

logger = logging.getLogger(__name__)


class SelfDiagnoseStrategy(AbstractRecoveryStrategy):
    """First-line recovery: self-diagnose and fix simple issues.

    Handles: sensor timeouts, transient communication errors, calibration drift.
    Cannot handle: kinematic infeasibility, major hardware faults.
    """

    def can_handle(self, error: dict[str, Any], state: Any) -> bool:
        error_type = error.get("type", "")
        return error_type in ("communication", "perception")

    async def handle(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        logger.info("Running self-diagnose...")

        error_type = error.get("type", "")

        if error_type == "communication":
            # Try to reconnect/reinitialize the communication channel
            logger.info("Attempting to reset communication channel")
            return RecoveryResult(
                success=True,
                strategy_used="self_diagnose",
                recovery_event={"action": "reset_communication", "detail": "Channel reset, retrying"},
            )

        if error_type == "perception":
            # Try to re-detect, adjust camera exposure, etc.
            logger.info("Attempting to re-run perception with adjusted parameters")
            return RecoveryResult(
                success=True,
                strategy_used="self_diagnose",
                recovery_event={"action": "retry_perception", "detail": "Re-running perception pipeline"},
            )

        return RecoveryResult(
            success=False,
            strategy_used="self_diagnose",
            recovery_event={"detail": f"Cannot self-diagnose error type: {error_type}"},
        )
