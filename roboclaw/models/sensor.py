"""Sensor data models for multi-modal perception."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraFrame(BaseModel):
    """Single RGB-D camera frame."""

    camera_id: str
    timestamp: float
    width: int = 640
    height: int = 480
    rgb_encoded: bytes | None = None  # JPEG/PNG encoded
    depth_encoded: bytes | None = None
    intrinsics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # fx, fy, cx, cy


class LidarScan(BaseModel):
    """2D or 3D LiDAR scan."""

    sensor_id: str
    timestamp: float
    frame_id: str = "lidar_link"
    points: list[tuple[float, float, float]] = Field(default_factory=list)  # N x (x, y, z)
    intensities: list[float] = Field(default_factory=list)
    is_3d: bool = True


class AudioChunk(BaseModel):
    """Raw audio chunk from microphone."""

    microphone_id: str
    timestamp: float
    sample_rate_hz: int = 16000
    samples: bytes = b""
    duration_s: float = 0.0


class IMUReading(BaseModel):
    """Inertial Measurement Unit reading."""

    sensor_id: str
    timestamp: float
    frame_id: str = "imu_link"
    linear_accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class ForceTorque(BaseModel):
    """6-axis force-torque sensor reading."""

    sensor_id: str = "wrist_ft"
    frame_id: str = "ee_link"
    timestamp: float = 0.0
    force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict) -> "ForceTorque":
        return cls(
            sensor_id=str(data.get("sensor_id", "wrist_ft")),
            frame_id=str(data.get("frame_id", "ee_link")),
            timestamp=float(data.get("timestamp", 0.0)),
            force_xyz=tuple(float(v) for v in data["force_xyz"]),
            torque_xyz=tuple(float(v) for v in data["torque_xyz"]),
        )

    def to_dict(self) -> dict:
        return self.model_dump()


class ProprioceptiveState(BaseModel):
    """Full proprioceptive state of the humanoid robot."""

    timestamp: float = 0.0
    joint_positions: dict[str, float] = Field(default_factory=dict)
    joint_velocities: dict[str, float] = Field(default_factory=dict)
    joint_torques: dict[str, float] = Field(default_factory=dict)
    imu: IMUReading | None = None
    left_foot_pressure: tuple[float, ...] = ()
    right_foot_pressure: tuple[float, ...] = ()
    left_arm_ft: ForceTorque | None = None
    right_arm_ft: ForceTorque | None = None
    cop_left: tuple[float, float] = (0.0, 0.0)
    cop_right: tuple[float, float] = (0.0, 0.0)
    zmp: tuple[float, float] = (0.0, 0.0)
