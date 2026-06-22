"""Navigation executor stub — path following for wheeled or legged base."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


class NavigationExecutor(AbstractActionExecutor):
    """Path-following executor for robot base navigation.

    Phase 1 stub — provides simulated path following.
    """

    def __init__(self) -> None:
        self._cancelled = False

    async def execute(self, sub_task: dict[str, Any], world_state: dict[str, Any]) -> ActionResult:
        self._cancelled = False
        params = sub_task.get("parameters", {})

        logger.info(f"Following path to {params} (simulated)")
        # Simulated execution
        import asyncio
        await asyncio.sleep(1.0)

        return ActionResult(
            sub_task_id=sub_task.get("sub_task_id", "navigation"),
            status=ActionResultStatus.SUCCESS,
            duration_sec=1.0,
            actual_outcome={"at_goal": True, "simulated": True},
        )

    async def cancel(self) -> None:
        self._cancelled = True

    async def emergency_stop(self) -> None:
        self._cancelled = True
