"""Skill definition and parameter models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SkillParameter(BaseModel):
    """A parameter accepted by a skill."""

    name: str
    param_type: str = "string"  # string | float | int | bool | pose | joint_array
    description: str = ""
    required: bool = True
    default: Any = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class SkillDefinition(BaseModel):
    """Definition of a parameterized robot skill."""

    name: str
    skill_type: str  # SkillType enum value
    description: str = ""
    parameters: list[SkillParameter] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    timeout_sec: float = 30.0
    applicable_embodiments: list[str] = Field(default_factory=list)  # empty = all
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
