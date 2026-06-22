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
