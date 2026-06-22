"""RoboClaw: Production-Grade Vertical Agent Framework for Humanoid Robotics.

A LangGraph-based agent framework implementing the Perceive→Reflect→Plan→Act→Feedback
loop for bipedal humanoid robots, with multi-tier memory, failure recovery,
and A2A multi-agent protocol support.

Architecture Layers:
    Perception → Planning → Memory → Action → Recovery → API → A2A
"""

from roboclaw.core.config import RobotConfig, Settings
from roboclaw.core.errors import RoboClawError
from roboclaw.core.types import AgentMode, AgentPhase, FailureType, Outcome, SkillType
from roboclaw.models.episode import EpisodeEvent, EpisodeRecord
from roboclaw.models.state import HumanoidState, RobotState
from roboclaw.orchestration.graph import build_agent_graph, create_agent_runner

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentMode",
    "AgentPhase",
    "FailureType",
    "Outcome",
    "RoboClawError",
    "RobotConfig",
    "Settings",
    "SkillType",
    # Models
    "EpisodeEvent",
    "EpisodeRecord",
    "HumanoidState",
    "RobotState",
    # Orchestration
    "build_agent_graph",
    "create_agent_runner",
]
