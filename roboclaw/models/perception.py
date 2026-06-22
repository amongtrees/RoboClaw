"""Perception result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    """An object detected in the scene."""

    label: str
    instance_id: str = ""
    bbox_2d: tuple[float, float, float, float] | None = None  # x, y, w, h
    mask: list[list[int]] | None = None
    position_world: tuple[float, float, float] | None = None
    orientation_world: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    attributes: dict[str, str] = Field(default_factory=dict)


class SceneGraph(BaseModel):
    """Symbolic scene graph representing spatial and semantic relationships."""

    objects: list[DetectedObject] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    rooms: list[str] = Field(default_factory=list)
    navigable_areas: list[dict[str, Any]] = Field(default_factory=list)


class PerceptionSnapshot(BaseModel):
    """Aggregated perception data for one processing cycle."""

    timestamp: float = 0.0
    rgb_frames: dict[str, bytes] = Field(default_factory=dict)  # camera_id -> encoded frame
    depth_frames: dict[str, bytes] = Field(default_factory=dict)
    audio_chunk: bytes | None = None
    joint_states: dict[str, float] = Field(default_factory=dict)
    imu_data: dict[str, Any] = Field(default_factory=dict)
    force_torque: dict[str, Any] = Field(default_factory=dict)
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    speech_text: str | None = None
    sound_events: list[str] = Field(default_factory=list)
    scene_graph: dict[str, Any] = Field(default_factory=dict)
