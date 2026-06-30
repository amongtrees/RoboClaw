"""ROS 2 action executors — drive real robot hardware via ROS 2.

These implement ``AbstractActionExecutor`` by translating RoboClaw
skill invocations into ROS 2 ``JointTrajectory`` messages and
publishing them through the ``ROS2Bridge``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time as time_mod
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------


def _build_walk_trajectory(
    joint_names: list[str],
    num_steps: int,
    direction: str = "forward",
    step_length: float = 0.3,
    num_waypoints_per_step: int = 20,
) -> list[dict[str, float]]:
    """Build a CPG-based walking trajectory for the given leg joints.

    Generates sinusoidal hip/knee/ankle trajectories that produce a
    bipedal walking gait.

    Parameters
    ----------
    joint_names:
        Ordered list of leg joint names (e.g. ``["left_hip_pitch", ...]``).
    num_steps:
        Number of foot-steps.
    direction:
        ``forward`` | ``backward``.
    step_length:
        Step length in meters (affects amplitude).
    num_waypoints_per_step:
        Control points per step.

    Returns
    -------
    list of dicts, each mapping joint_name → position (rad).
    """
    total_waypoints = num_steps * num_waypoints_per_step
    hip_amp = 0.3 * min(step_length / 0.3, 1.0)
    knee_amp = 0.5 * min(step_length / 0.3, 1.0)
    ankle_amp = 0.15 * min(step_length / 0.3, 1.0)

    if direction == "backward":
        hip_amp *= -1.0

    waypoints: list[dict[str, float]] = []
    for t in range(total_waypoints):
        phase = 2.0 * math.pi * t / num_waypoints_per_step
        wp: dict[str, float] = {}

        for name in joint_names:
            val = 0.0

            # Left leg (phase 0)
            if "left_hip_pitch" in name:
                val = hip_amp * math.sin(phase)
            elif "left_knee" in name:
                val = knee_amp * max(0.0, math.sin(phase))
            elif "left_ankle" in name:
                val = ankle_amp * math.sin(phase)

            # Right leg (phase π — alternating)
            elif "right_hip_pitch" in name:
                val = hip_amp * math.sin(phase + math.pi)
            elif "right_knee" in name:
                val = knee_amp * max(0.0, math.sin(phase + math.pi))
            elif "right_ankle" in name:
                val = ankle_amp * math.sin(phase + math.pi)

            wp[name] = val

        waypoints.append(wp)

    return waypoints


def _build_reach_trajectory(
    joint_names: list[str],
    arm: str,
    target_height: float = 0.8,
    target_reach: float = 0.4,
    num_waypoints: int = 50,
) -> list[dict[str, float]]:
    """Build a reaching trajectory for one arm.

    Uses linear interpolation from zero to a target shoulder/elbow
    configuration.
    """
    waypoints: list[dict[str, float]] = []
    shoulder_target = -0.5 if arm == "right" else 0.5
    elbow_target = -1.2

    for t in range(num_waypoints):
        alpha = (t + 1) / num_waypoints
        wp: dict[str, float] = {}
        for name in joint_names:
            val = 0.0
            if arm in name:
                if "shoulder_pitch" in name:
                    val = alpha * shoulder_target
                elif "elbow" in name:
                    val = alpha * elbow_target
            wp[name] = val
        waypoints.append(wp)

    return waypoints


def _build_climb_trajectory(
    joint_names: list[str],
    num_stairs: int,
    direction: str = "up",
    num_waypoints_per_stair: int = 30,
) -> list[dict[str, float]]:
    """Build a stair-climbing trajectory with higher knee lift."""
    total = num_stairs * num_waypoints_per_stair
    hip_amp = 0.5
    knee_amp = 0.8
    if direction == "down":
        hip_amp *= -1.0
        knee_amp *= 0.6

    waypoints: list[dict[str, float]] = []
    for t in range(total):
        phase = 2.0 * math.pi * t / num_waypoints_per_stair
        wp: dict[str, float] = {}
        for name in joint_names:
            val = 0.0
            if "left_hip_pitch" in name:
                val = hip_amp * math.sin(phase)
            elif "left_knee" in name:
                val = knee_amp * max(0.0, math.sin(phase))
            elif "left_ankle" in name:
                val = 0.2 * math.sin(phase)
            elif "right_hip_pitch" in name:
                val = hip_amp * math.sin(phase + math.pi)
            elif "right_knee" in name:
                val = knee_amp * max(0.0, math.sin(phase + math.pi))
            elif "right_ankle" in name:
                val = 0.2 * math.sin(phase + math.pi)
            wp[name] = val
        waypoints.append(wp)

    return waypoints


# ---------------------------------------------------------------------------
# ROS2LocomotionExecutor
# ---------------------------------------------------------------------------


class ROS2LocomotionExecutor(AbstractActionExecutor):
    """Execute locomotion skills by publishing JointTrajectory via ROS 2.

    Skills: navigate_to, walk_steps, climb_stairs.

    Parameters
    ----------
    bridge:
        The ROS2Bridge for publishing.
    joint_names:
        Ordered list of leg joint names for trajectory generation.
    duration_per_point:
        Seconds per trajectory waypoint.
    """

    def __init__(
        self,
        bridge: Any,
        joint_names: list[str] | None = None,
        duration_per_point: float = 0.05,
        robot_config: Any = None,
    ) -> None:
        self._bridge = bridge
        self._cancelled = False
        self._duration_per_point = duration_per_point
        self._config = robot_config

        # Default H1 leg joint names if none provided
        self._joint_names = joint_names or [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "torso",
        ]

    async def execute(
        self, sub_task: dict[str, Any], world_state: dict[str, Any]
    ) -> ActionResult:
        self._cancelled = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "")
        sub_task_id = sub_task.get("sub_task_id", skill_type)

        try:
            if skill_type == "walk_steps":
                return await self._walk_steps(params, sub_task_id)
            elif skill_type == "navigate_to":
                return await self._navigate_to(params, sub_task_id)
            elif skill_type == "climb_stairs":
                return await self._climb_stairs(params, sub_task_id)
            else:
                return ActionResult(
                    sub_task_id=sub_task_id,
                    status=ActionResultStatus.FAILURE,
                    error={
                        "type": "unknown_skill",
                        "detail": f"ROS2 locomotion: '{skill_type}' not supported",
                    },
                )
        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled"},
            )

    async def _walk_steps(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        num_steps = int(params.get("num_steps", 1))
        direction = str(params.get("direction", "forward"))
        step_length = float(params.get("step_length_m", 0.3))

        waypoints = _build_walk_trajectory(
            self._joint_names, num_steps, direction, step_length
        )

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "walk_steps",
                "params": params,
                "waypoints": len(waypoints),
                "duration_sec": duration,
            },
        )

    async def _navigate_to(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        room = params.get("room", "unknown")
        # Approximate: 10 steps per 5 meters
        distance = 5.0
        if isinstance(room, str) and room.lower() in ("kitchen", "living_room"):
            distance = 8.0
        num_steps = max(1, int(distance / 0.3))

        waypoints = _build_walk_trajectory(self._joint_names, num_steps, "forward", 0.3)

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "navigate_to",
                "room": room,
                "waypoints": len(waypoints),
            },
        )

    async def _climb_stairs(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        num_stairs = int(params.get("num_steps", 5))
        direction = str(params.get("direction", "up"))

        waypoints = _build_climb_trajectory(self._joint_names, num_stairs, direction)

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "climb_stairs",
                "waypoints": len(waypoints),
            },
        )

    async def cancel(self) -> None:
        self._cancelled = True
        logger.info("ROS2LocomotionExecutor cancelled")

    async def emergency_stop(self) -> None:
        self._cancelled = True
        logger.warning("ROS2LocomotionExecutor emergency stop!")


# ---------------------------------------------------------------------------
# ROS2ManipulationExecutor
# ---------------------------------------------------------------------------


class ROS2ManipulationExecutor(AbstractActionExecutor):
    """Execute manipulation skills by publishing arm JointTrajectory via ROS 2.

    Skills: whole_body_grasp, place_object, handover, open_door.

    Parameters
    ----------
    bridge:
        The ROS2Bridge for publishing.
    arm_joint_names:
        Ordered list of arm joint names.
    duration_per_point:
        Seconds per trajectory waypoint.
    """

    def __init__(
        self,
        bridge: Any,
        arm_joint_names: list[str] | None = None,
        duration_per_point: float = 0.05,
        robot_config: Any = None,
    ) -> None:
        self._bridge = bridge
        self._cancelled = False
        self._duration_per_point = duration_per_point
        self._config = robot_config

        # Default H1 arm joint names
        self._arm_joint_names = arm_joint_names or [
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        ]

    async def execute(
        self, sub_task: dict[str, Any], world_state: dict[str, Any]
    ) -> ActionResult:
        self._cancelled = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "")
        sub_task_id = sub_task.get("sub_task_id", skill_type)

        try:
            if skill_type in ("whole_body_grasp",):
                return await self._grasp(params, sub_task_id)
            elif skill_type == "place_object":
                return await self._place(params, sub_task_id)
            elif skill_type == "handover":
                return await self._handover(params, sub_task_id)
            elif skill_type == "open_door":
                return await self._open_door(params, sub_task_id)
            else:
                return ActionResult(
                    sub_task_id=sub_task_id,
                    status=ActionResultStatus.FAILURE,
                    error={
                        "type": "unknown_skill",
                        "detail": f"ROS2 manipulation: '{skill_type}' not supported",
                    },
                )
        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled"},
            )

    async def _grasp(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        arm = str(params.get("arm", "right"))
        waypoints = _build_reach_trajectory(self._arm_joint_names, arm, num_waypoints=50)

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._arm_joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "whole_body_grasp",
                "params": params,
                "waypoints": len(waypoints),
            },
        )

    async def _place(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        arm = str(params.get("arm", "right"))
        waypoints = _build_reach_trajectory(
            self._arm_joint_names, arm, target_height=0.5, num_waypoints=40
        )

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._arm_joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "place_object",
                "params": params,
                "waypoints": len(waypoints),
            },
        )

    async def _handover(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        arm = str(params.get("arm", "right"))
        waypoints = _build_reach_trajectory(self._arm_joint_names, arm, num_waypoints=60)

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._arm_joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "handover",
                "params": params,
                "waypoints": len(waypoints),
            },
        )

    async def _open_door(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        arm = str(params.get("arm", "right"))
        waypoints = _build_reach_trajectory(self._arm_joint_names, arm, num_waypoints=60)

        start = time_mod.time()
        await self._bridge.publish_trajectory_from_dict(
            self._arm_joint_names, waypoints, self._duration_per_point
        )
        duration = time_mod.time() - start

        return ActionResult(
            sub_task_id=sub_task_id,
            status=ActionResultStatus.SUCCESS,
            duration_sec=duration,
            actual_outcome={
                "simulated": False,
                "backend": "ros2",
                "skill": "open_door",
                "params": params,
                "waypoints": len(waypoints),
            },
        )

    async def cancel(self) -> None:
        self._cancelled = True
        logger.info("ROS2ManipulationExecutor cancelled")

    async def emergency_stop(self) -> None:
        self._cancelled = True
        logger.warning("ROS2ManipulationExecutor emergency stop!")
