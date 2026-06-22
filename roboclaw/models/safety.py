"""Safety-related models."""

from __future__ import annotations

from time import time
from typing import Any

from pydantic import BaseModel, Field

from roboclaw.core.types import SafetySeverity


class SafetyZone(BaseModel):
    """A 3D safety zone (keep-out region)."""

    zone_id: str
    zone_type: str  # keep_out | slow_down | warned
    geometry_type: str = "box"  # box | sphere | cylinder
    dimensions: dict[str, float] = Field(default_factory=dict)  # e.g., {"x":1, "y":1, "z":2}
    position_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_world: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class ForceLimit(BaseModel):
    """Force/torque limit for a joint or end-effector."""

    component: str  # joint name or "left_gripper", "right_arm"
    max_absolute_force_n: float = 100.0
    max_absolute_torque_nm: float = 50.0
    soft_limit_ratio: float = 0.8  # warning at 80% of max


class SafetyViolation(BaseModel):
    """Record of a safety constraint violation."""

    event_id: str = ""
    robot_id: str = ""
    violation_type: str = ""  # force_limit | joint_limit | workspace_boundary | zmp_violation | collision
    severity: SafetySeverity = SafetySeverity.WARNING
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time)
    resolved: bool = False
