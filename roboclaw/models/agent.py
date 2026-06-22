"""Agent models for A2A protocol and multi-agent systems."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roboclaw.models.skill import SkillDefinition


class AgentCapability(BaseModel):
    """A capability advertised by an agent."""

    capability_id: str
    description: str
    embedding: list[float] | None = None  # For semantic matching


class AgentCard(BaseModel):
    """Agent Card per the A2A protocol spec — advertises an agent's capabilities."""

    name: str
    description: str = ""
    url: str = ""  # HTTP endpoint
    version: str = "0.1.0"
    capabilities: list[AgentCapability] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])
    skills: list[SkillDefinition] = Field(default_factory=list)
    provider: str = ""
    documentation_url: str = ""


class AgentMessage(BaseModel):
    """A message exchanged between agents (A2A or internal)."""

    message_id: str = ""
    sender_agent: str = ""
    receiver_agent: str = ""
    message_type: str = "task_request"  # task_request | task_response | status_update | handoff
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0
    reply_to: str | None = None
    correlation_id: str | None = None
