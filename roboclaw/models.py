"""Core data models for RoboClaw memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Literal
from uuid import uuid4


Outcome = Literal["success", "failure", "blocked", "aborted", "unknown"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class Pose:
    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pose":
        return cls(
            frame_id=str(data["frame_id"]),
            position=tuple(float(v) for v in data["position"]),
            orientation_xyzw=tuple(float(v) for v in data["orientation_xyzw"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ForceTorque:
    frame_id: str
    force_xyz: tuple[float, float, float]
    torque_xyz: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForceTorque":
        return cls(
            frame_id=str(data["frame_id"]),
            force_xyz=tuple(float(v) for v in data["force_xyz"]),
            torque_xyz=tuple(float(v) for v in data["torque_xyz"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RobotState:
    robot_id: str
    timestamp: float = field(default_factory=time)
    frame_id: str = "base_link"
    camera_frame_id: str | None = None
    task_id: str | None = None
    skill_name: str | None = None
    phase: str | None = None
    joint_states: dict[str, float] = field(default_factory=dict)
    end_effector_pose: Pose | None = None
    gripper_state: dict[str, Any] = field(default_factory=dict)
    force_torque: ForceTorque | None = None
    visible_objects: list[dict[str, Any]] = field(default_factory=list)
    safety_state: dict[str, Any] = field(default_factory=dict)
    tf_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    uri: str
    episode_id: str | None = None
    created_at: float = field(default_factory=time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        artifact_type: str,
        uri: str,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactRef":
        return cls(
            artifact_id=new_id("artifact"),
            artifact_type=artifact_type,
            uri=uri,
            episode_id=episode_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeEvent:
    event_id: str
    episode_id: str
    event_type: str
    timestamp: float = field(default_factory=time)
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        episode_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> "EpisodeEvent":
        return cls(
            event_id=new_id("event"),
            episode_id=episode_id,
            event_type=event_type,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    robot_id: str
    task_type: str
    goal: str
    skill_name: str | None = None
    started_at: float = field(default_factory=time)
    ended_at: float | None = None
    outcome: Outcome = "unknown"
    summary: str = ""
    initial_state: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        robot_id: str,
        task_type: str,
        goal: str,
        skill_name: str | None = None,
        initial_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EpisodeRecord":
        return cls(
            episode_id=new_id("episode"),
            robot_id=robot_id,
            task_type=task_type,
            goal=goal,
            skill_name=skill_name,
            initial_state=initial_state or {},
            metadata=metadata or {},
        )

    def searchable_text(self) -> str:
        parts = [
            self.robot_id,
            self.task_type,
            self.goal,
            self.skill_name or "",
            self.outcome,
            self.summary,
            " ".join(str(v) for v in self.metadata.values()),
        ]
        return " ".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
