"""Episode recording models — the event-sourced execution history."""

from __future__ import annotations

from time import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Outcome = Literal["success", "failure", "blocked", "aborted", "unknown"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Artifact(BaseModel):
    """Reference to a stored artifact (trajectory, video, point cloud, etc.)."""

    artifact_id: str = Field(default_factory=lambda: new_id("artifact"))
    artifact_type: str  # trajectory_jsonl | rosbag | pointcloud | video
    uri: str
    episode_id: str | None = None
    created_at: float = Field(default_factory=time)
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        artifact_type: str,
        uri: str,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Artifact":
        return cls(
            artifact_type=artifact_type,
            uri=uri,
            episode_id=episode_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        return self.model_dump()


class EpisodeEvent(BaseModel):
    """A single event within an episode timeline."""

    event_id: str = Field(default_factory=lambda: new_id("event"))
    episode_id: str
    event_type: str
    timestamp: float = Field(default_factory=time)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        episode_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> "EpisodeEvent":
        return cls(
            episode_id=episode_id,
            event_type=event_type,
            payload=payload or {},
        )

    def to_dict(self) -> dict:
        return self.model_dump()


class EpisodeRecord(BaseModel):
    """Full record of a task execution episode."""

    episode_id: str = Field(default_factory=lambda: new_id("episode"))
    robot_id: str
    task_id: str | None = None
    task_type: str
    goal: str
    skill_name: str | None = None
    plan_id: str | None = None
    started_at: float = Field(default_factory=time)
    ended_at: float | None = None
    outcome: Outcome = "unknown"
    summary: str = ""
    initial_state: dict[str, Any] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_attempts: int = 0
    total_duration_sec: float | None = None

    @classmethod
    def create(
        cls,
        robot_id: str,
        task_type: str,
        goal: str,
        skill_name: str | None = None,
        task_id: str | None = None,
        initial_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EpisodeRecord":
        return cls(
            robot_id=robot_id,
            task_type=task_type,
            goal=goal,
            skill_name=skill_name,
            task_id=task_id,
            initial_state=initial_state or {},
            metadata=metadata or {},
        )

    def finalize(self, outcome: Outcome, summary: str = "", final_state: dict | None = None) -> None:
        """Mark the episode as complete."""
        self.outcome = outcome
        self.summary = summary
        self.ended_at = time()
        if final_state:
            self.final_state = final_state
        if self.ended_at and self.started_at:
            self.total_duration_sec = self.ended_at - self.started_at

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

    def to_dict(self) -> dict:
        return self.model_dump()
