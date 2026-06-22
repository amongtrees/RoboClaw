"""Pose, twist, and gait parameter models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Pose(BaseModel):
    """3D pose with frame reference."""

    frame_id: str = "base_link"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict) -> "Pose":
        return cls(
            frame_id=str(data.get("frame_id", "base_link")),
            position=tuple(float(v) for v in data["position"]),
            orientation_xyzw=tuple(float(v) for v in data["orientation_xyzw"]),
        )

    def to_dict(self) -> dict:
        return self.model_dump()


class Twist(BaseModel):
    """6-DOF velocity (linear + angular)."""

    frame_id: str = "base_link"
    linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)


class BipedalGaitParams(BaseModel):
    """Parameters for bipedal walking gait generation."""

    step_length_m: float = 0.3
    step_width_m: float = 0.15
    step_height_m: float = 0.05
    step_duration_s: float = 0.6
    double_support_ratio: float = 0.2
    swing_height_m: float = 0.08
    trunk_pitch_rad: float = 0.0
    trunk_roll_rad: float = 0.0
    com_height_m: float = 0.85
