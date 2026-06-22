"""Robot state models — the core state representation."""

from __future__ import annotations

from time import time
from typing import Any

from pydantic import BaseModel, Field

from roboclaw.models.pose import Pose
from roboclaw.models.sensor import ForceTorque


class RobotState(BaseModel):
    """General robot state (cross-embodiment)."""

    robot_id: str
    timestamp: float = Field(default_factory=time)
    frame_id: str = "base_link"
    camera_frame_id: str | None = None
    task_id: str | None = None
    skill_name: str | None = None
    phase: str | None = None
    joint_states: dict[str, float] = Field(default_factory=dict)
    end_effector_pose: Pose | None = None
    gripper_state: dict[str, Any] = Field(default_factory=dict)
    force_torque: ForceTorque | None = None
    visible_objects: list[dict[str, Any]] = Field(default_factory=list)
    safety_state: dict[str, Any] = Field(default_factory=dict)
    tf_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()


class HumanoidState(RobotState):
    """Humanoid-specific state including bipedal balance metrics."""

    # Dual-arm
    end_effector_pose_left: Pose | None = None
    end_effector_pose_right: Pose | None = None
    gripper_state_left: dict[str, Any] = Field(default_factory=dict)
    gripper_state_right: dict[str, Any] = Field(default_factory=dict)
    force_torque_left: ForceTorque | None = None
    force_torque_right: ForceTorque | None = None

    # Bipedal balance
    zmp: tuple[float, float] = (0.0, 0.0)
    cop_left: tuple[float, float] = (0.0, 0.0)
    cop_right: tuple[float, float] = (0.0, 0.0)
    left_foot_pose: Pose | None = None
    right_foot_pose: Pose | None = None
    foot_contact_left: bool = True
    foot_contact_right: bool = True
    support_phase: str = "double"  # double | left_single | right_single
    com_position: tuple[float, float, float] = (0.0, 0.0, 0.85)
    com_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Head
    head_pan_rad: float = 0.0
    head_tilt_rad: float = 0.0
