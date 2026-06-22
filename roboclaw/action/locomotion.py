"""Bipedal locomotion executor stub.

In production, this interfaces with a whole-body controller (ZMP preview,
MPC, or DDP-based) via ROS 2 joint trajectory commands.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


class LocomotionExecutor(AbstractActionExecutor):
    """Executor for bipedal walking and balance control.

    Phase 1 stub — simulates walking with timing delays.
    """

    def __init__(self, robot_config: Any = None) -> None:
        self._config = robot_config
        self._cancelled = False

    async def execute(self, sub_task: dict[str, Any], world_state: dict[str, Any]) -> ActionResult:
        """Execute a locomotion sub-task (navigate_to, walk_steps, climb_stairs)."""
        self._cancelled = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "")

        try:
            if skill_type == "navigate_to":
                return await self._navigate_to(params)
            elif skill_type == "walk_steps":
                return await self._walk_steps(params)
            elif skill_type == "climb_stairs":
                return await self._climb_stairs(params)
            else:
                return ActionResult(
                    sub_task_id=sub_task.get("sub_task_id", ""),
                    status=ActionResultStatus.FAILURE,
                    error={"type": "unknown_skill", "detail": f"Locomotion skill '{skill_type}' not supported"},
                )
        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task.get("sub_task_id", ""),
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled", "detail": "Locomotion was cancelled"},
            )

    async def _navigate_to(self, params: dict[str, Any]) -> ActionResult:
        """Simulate walking to a room."""
        room = params.get("room", "unknown")
        duration = 5.0  # simulated walking time
        logger.info(f"Walking to room: {room} (simulated {duration}s)")

        for i in range(int(duration)):
            if self._cancelled:
                break
            await asyncio.sleep(0.1)

        return ActionResult(
            sub_task_id="navigate_to",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"at_location": room, "simulated": True},
        )

    async def _walk_steps(self, params: dict[str, Any]) -> ActionResult:
        """Simulate walking a number of steps."""
        num_steps = params.get("num_steps", 1)
        step_duration = 0.6
        duration = num_steps * step_duration

        logger.info(f"Walking {num_steps} steps (simulated {duration:.1f}s)")
        await asyncio.sleep(min(duration, 2.0))  # cap simulation time

        return ActionResult(
            sub_task_id="walk_steps",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"steps_completed": num_steps, "simulated": True},
        )

    async def _climb_stairs(self, params: dict[str, Any]) -> ActionResult:
        """Simulate climbing stairs."""
        num_steps = params.get("num_steps", 5)
        direction = params.get("direction", "up")
        duration = num_steps * 0.8

        logger.info(f"Climbing {num_steps} stairs {direction} (simulated {duration:.1f}s)")
        await asyncio.sleep(min(duration, 2.0))

        return ActionResult(
            sub_task_id="climb_stairs",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"stairs_climbed": num_steps, "direction": direction, "simulated": True},
        )

    async def cancel(self) -> None:
        """Graceful stop — complete current step, then halt."""
        self._cancelled = True
        logger.info("Locomotion cancelled gracefully")

    async def emergency_stop(self) -> None:
        """Immediate hard stop."""
        self._cancelled = True
        logger.warning("Locomotion emergency stop!")
