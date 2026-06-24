"""Shared enums and type aliases for the RoboClaw framework."""

from __future__ import annotations

from enum import StrEnum


class AgentPhase(StrEnum):
    """Phases of the main agent loop."""

    IDLE = "idle"
    PERCEIVING = "perceiving"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    ACTING = "acting"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"
    DELEGATING = "delegating"
    HRI = "hri"


class Outcome(StrEnum):
    """Outcome of an episode or sub-task."""

    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class SkillType(StrEnum):
    """Catalog of humanoid robot skills."""

    NAVIGATE_TO = "navigate_to"
    WALK_STEPS = "walk_steps"
    WHOLE_BODY_GRASP = "whole_body_grasp"
    BI_MANUAL_CARRY = "bi_manual_carry"
    PLACE_OBJECT = "place_object"
    HANDOVER = "handover"
    SPEAK = "speak"
    GAZE_AT = "gaze_at"
    WAIT_FOR = "wait_for"
    OPEN_DOOR = "open_door"
    CLIMB_STAIRS = "climb_stairs"
    PUSH_OBJECT = "push_object"
    PULL_OBJECT = "pull_object"
    PRESS_BUTTON = "press_button"


class FailureType(StrEnum):
    """Classification of execution failures."""

    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"
    PERCEPTION = "perception"
    PLANNING = "planning"
    SAFETY = "safety"
    COMMUNICATION = "communication"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


class AgentMode(StrEnum):
    """Operating mode of a robot agent."""

    IDLE = "idle"
    AUTONOMOUS = "autonomous"
    TELEOP = "teleop"
    DIAGNOSTIC = "diagnostic"
    EMERGENCY_STOP = "emergency_stop"


class SafetySeverity(StrEnum):
    """Severity level of safety events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    E_STOP = "e_stop"


class ModelType(StrEnum):
    """Which specialized foundation model handles a given subtask.

    Maps to the Qwen-RobotSuite tripartite architecture:
      - VLN (Vision-Language Navigation): handles navigation / locomotion
      - VLA (Vision-Language-Action): handles manipulation / grasping
      - WORLD_MODEL: predictive physical simulation for lookahead safety
      - LLM: generic language reasoning (planning, reflection, HRI)
      - NONE: pure rule-based or simulation stub (no model needed)
    """

    VLN = "vln"
    VLA = "vla"
    WORLD_MODEL = "world_model"
    LLM = "llm"
    NONE = "none"


# Mapping from SkillType to the ModelType domain that should execute it.
_SKILL_MODEL_MAP: dict[str, str] = {
    # --- VLN domain: locomotion & navigation ---
    SkillType.NAVIGATE_TO: "vln",
    SkillType.WALK_STEPS: "vln",
    SkillType.CLIMB_STAIRS: "vln",
    # --- VLA domain: manipulation & grasping ---
    SkillType.WHOLE_BODY_GRASP: "vla",
    SkillType.BI_MANUAL_CARRY: "vla",
    SkillType.PLACE_OBJECT: "vla",
    SkillType.HANDOVER: "vla",
    SkillType.OPEN_DOOR: "vla",
    SkillType.PUSH_OBJECT: "vla",
    SkillType.PULL_OBJECT: "vla",
    SkillType.PRESS_BUTTON: "vla",
    # --- LLM / rule-based ---
    SkillType.SPEAK: "llm",
    SkillType.GAZE_AT: "llm",
    SkillType.WAIT_FOR: "llm",
}


def model_type_for_skill(skill_type: str) -> str:
    """Return the ModelType value that should execute a given SkillType."""
    return _SKILL_MODEL_MAP.get(skill_type, "llm")
