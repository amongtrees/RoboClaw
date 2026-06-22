"""Head/neck control executor stub — active gaze and camera orientation."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


class HeadNeckExecutor(AbstractActionExecutor):
    """Controls head pan/tilt for active camera gaze.

    Phase 1 stub.
    """

    def __init__(self) -> None:
        self._cancelled = False

    async def execute(self, sub_task: dict[str, Any], world_state: dict[str, Any]) -> ActionResult:
        params = sub_task.get("parameters", {})
        target = params.get("target", "scene")

        logger.info(f"Gazing at: {target} (simulated)")

        return ActionResult(
            sub_task_id=sub_task.get("sub_task_id", "gaze"),
            status=ActionResultStatus.SUCCESS,
            duration_sec=0.5,
            actual_outcome={"gazed_at": target, "simulated": True},
        )

    async def cancel(self) -> None:
        self._cancelled = True

    async def emergency_stop(self) -> None:
        self._cancelled = True
