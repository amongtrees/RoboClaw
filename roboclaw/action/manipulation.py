"""Dual-arm manipulation executor stub.

In production, this drives arm joint trajectories, gripper control,
and force-torque monitoring via ROS 2 action servers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


class ManipulationExecutor(AbstractActionExecutor):
    """Executor for dual-arm manipulation tasks.

    Phase 1 stub — simulates grasp, place, and handover actions.
    """

    def __init__(self, robot_config: Any = None) -> None:
        self._config = robot_config
        self._cancelled = False

    async def execute(self, sub_task: dict[str, Any], world_state: dict[str, Any]) -> ActionResult:
        """Execute a manipulation sub-task."""
        self._cancelled = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "")

        try:
            if skill_type in ("whole_body_grasp",):
                return await self._grasp(params)
            elif skill_type == "place_object":
                return await self._place(params)
            elif skill_type == "handover":
                return await self._handover(params)
            elif skill_type == "open_door":
                return await self._open_door(params)
            else:
                return ActionResult(
                    sub_task_id=sub_task.get("sub_task_id", ""),
                    status=ActionResultStatus.FAILURE,
                    error={"type": "unknown_skill", "detail": f"Manipulation skill '{skill_type}' not supported"},
                )
        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task.get("sub_task_id", ""),
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled"},
            )

    async def _grasp(self, params: dict[str, Any]) -> ActionResult:
        """Simulate a whole-body grasp."""
        obj = params.get("object", "unknown")
        arm = params.get("arm", "right")
        grasp_type = params.get("grasp_type", "top_down_encompassing")
        duration = 3.0

        logger.info(f"Grasping '{obj}' with {arm} arm, strategy={grasp_type} (simulated)")
        await asyncio.sleep(1.0)  # simulated execution

        return ActionResult(
            sub_task_id="whole_body_grasp",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"grasped": True, "object": obj, "arm": arm, "grasp_type": grasp_type, "simulated": True},
        )

    async def _place(self, params: dict[str, Any]) -> ActionResult:
        """Simulate placing an object."""
        obj = params.get("object", "unknown")
        target = (params.get("target_x", 0), params.get("target_y", 0), params.get("target_z", 0))
        duration = 2.0

        logger.info(f"Placing '{obj}' at {target} (simulated)")
        await asyncio.sleep(0.5)

        return ActionResult(
            sub_task_id="place_object",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"placed": True, "object": obj, "target": target, "simulated": True},
        )

    async def _handover(self, params: dict[str, Any]) -> ActionResult:
        """Simulate handing an object to a human or agent."""
        obj = params.get("object", "unknown")
        receiver = params.get("receiver", "human")
        duration = 4.0

        logger.info(f"Handing '{obj}' to {receiver} (simulated)")
        await asyncio.sleep(1.0)

        return ActionResult(
            sub_task_id="handover",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"handed_over": True, "object": obj, "receiver": receiver, "simulated": True},
        )

    async def _open_door(self, params: dict[str, Any]) -> ActionResult:
        """Simulate opening a door."""
        door = params.get("door", "unknown")
        arm = params.get("arm", "right")
        duration = 6.0

        logger.info(f"Opening door '{door}' with {arm} arm (simulated)")
        await asyncio.sleep(1.0)

        return ActionResult(
            sub_task_id="open_door",
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={"door_opened": True, "door": door, "simulated": True},
        )

    async def cancel(self) -> None:
        self._cancelled = True
        logger.info("Manipulation cancelled gracefully")

    async def emergency_stop(self) -> None:
        self._cancelled = True
        logger.warning("Manipulation emergency stop!")
