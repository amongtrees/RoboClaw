"""A2A protocol data models (Google A2A spec aligned)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roboclaw.models.skill import SkillDefinition


class AgentCapability(BaseModel):
    """A capability advertised by an agent."""

    capability_id: str
    description: str = ""
    embedding: list[float] | None = None


class AgentCard(BaseModel):
    """Agent Card — advertises an agent's capabilities per A2A spec."""

    name: str
    description: str = ""
    url: str = ""
    version: str = "0.1.0"
    capabilities: list[AgentCapability] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])
    skills: list[SkillDefinition] = Field(default_factory=list)
    provider: str = ""
    documentation_url: str = ""


class TaskDescriptor(BaseModel):
    """A task sent to a peer agent."""

    task_id: str = ""
    description: str = ""
    input_artifacts: list[str] = Field(default_factory=list)
    deadline_sec: float | None = None
    origin_agent: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    """Output of a completed A2A task."""

    artifact_id: str
    task_id: str = ""
    artifact_type: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: float = 0.0
