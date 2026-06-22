"""RoboClaw core layer: configuration, types, errors, and component registry."""

from roboclaw.core.config import RobotConfig, Settings
from roboclaw.core.errors import RoboClawError
from roboclaw.core.registry import ComponentRegistry
from roboclaw.core.types import AgentPhase, FailureType, Outcome, SkillType

__all__ = [
    "AgentPhase",
    "ComponentRegistry",
    "FailureType",
    "Outcome",
    "RoboClawError",
    "RobotConfig",
    "Settings",
    "SkillType",
]
