"""Domain error hierarchy for RoboClaw."""

from __future__ import annotations


class RoboClawError(Exception):
    """Base exception for all RoboClaw errors."""

    def __init__(self, message: str = "", *, code: str = "UNKNOWN", detail: dict | None = None):
        self.message = message
        self.code = code
        self.detail = detail or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


# --- Perception Errors ---


class PerceptionError(RoboClawError):
    """Error during perception processing."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="PERCEPTION_ERROR", **kwargs)


class SensorTimeoutError(PerceptionError):
    """Sensor data not received within expected interval."""

    def __init__(self, sensor_name: str, timeout_s: float):
        super().__init__(
            f"Sensor '{sensor_name}' timed out after {timeout_s}s",
            code="SENSOR_TIMEOUT",
            detail={"sensor_name": sensor_name, "timeout_s": timeout_s},
        )


class ObjectNotFoundError(PerceptionError):
    """Target object not found in scene."""

    def __init__(self, object_label: str):
        super().__init__(
            f"Object '{object_label}' not found in current scene",
            code="OBJECT_NOT_FOUND",
            detail={"object_label": object_label},
        )


# --- Planning Errors ---


class PlanningError(RoboClawError):
    """Error during task/motion planning."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="PLANNING_ERROR", **kwargs)


class NoFeasiblePlanError(PlanningError):
    """No feasible plan could be generated."""

    def __init__(self, goal: str, reason: str = ""):
        super().__init__(
            f"No feasible plan for goal: '{goal}'" + (f" — {reason}" if reason else ""),
            code="NO_FEASIBLE_PLAN",
            detail={"goal": goal, "reason": reason},
        )


class PlanValidationError(PlanningError):
    """Generated plan failed validation checks."""

    def __init__(self, plan_id: str, violations: list[str]):
        super().__init__(
            f"Plan '{plan_id}' validation failed: {'; '.join(violations)}",
            code="PLAN_VALIDATION_FAILED",
            detail={"plan_id": plan_id, "violations": violations},
        )


# --- Action/Execution Errors ---


class ActionError(RoboClawError):
    """Error during action execution."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="ACTION_ERROR", **kwargs)


class KinematicError(ActionError):
    """IK solution not found or joint limit violation."""

    def __init__(self, joint_name: str = "", limit: float = 0.0, actual: float = 0.0):
        super().__init__(
            f"Kinematic error on '{joint_name}': limit={limit}, actual={actual}",
            code="KINEMATIC_ERROR",
            detail={"joint_name": joint_name, "limit": limit, "actual": actual},
        )


class GraspFailedError(ActionError):
    """Grasp attempt did not achieve stable contact."""

    def __init__(self, object_label: str, force_n: float = 0.0):
        super().__init__(
            f"Grasp failed on '{object_label}' (contact force: {force_n}N)",
            code="GRASP_FAILED",
            detail={"object_label": object_label, "force_n": force_n},
        )


class CollisionError(ActionError):
    """Unexpected collision detected during execution."""

    def __init__(self, body_part: str, force_n: float):
        super().__init__(
            f"Collision on '{body_part}' with force {force_n}N",
            code="COLLISION_DETECTED",
            detail={"body_part": body_part, "force_n": force_n},
        )


# --- Safety Errors ---


class SafetyViolationError(RoboClawError):
    """Safety constraint violated."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="SAFETY_VIOLATION", **kwargs)


class ZMPViolationError(SafetyViolationError):
    """Zero Moment Point outside support polygon."""

    def __init__(self, zmp_x: float, zmp_y: float):
        super().__init__(
            f"ZMP ({zmp_x:.3f}, {zmp_y:.3f}) outside support polygon",
            code="ZMP_VIOLATION",
            detail={"zmp_x": zmp_x, "zmp_y": zmp_y},
        )


class ForceLimitExceededError(SafetyViolationError):
    """Joint torque or contact force exceeded safety limit."""

    def __init__(self, component: str, limit_nm: float, actual_nm: float):
        super().__init__(
            f"Force limit exceeded on '{component}': {actual_nm:.1f}Nm > {limit_nm:.1f}Nm",
            code="FORCE_LIMIT_EXCEEDED",
            detail={"component": component, "limit_nm": limit_nm, "actual_nm": actual_nm},
        )


# --- Communication Errors ---


class CommunicationError(RoboClawError):
    """Error in inter-service or robot communication."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="COMMUNICATION_ERROR", **kwargs)


class ROS2NodeError(CommunicationError):
    """ROS 2 node failure."""

    def __init__(self, node_name: str, reason: str = ""):
        super().__init__(
            f"ROS 2 node '{node_name}' error" + (f": {reason}" if reason else ""),
            code="ROS2_NODE_ERROR",
            detail={"node_name": node_name, "reason": reason},
        )


# --- Memory Errors ---


class MemoryError(RoboClawError):
    """Error in memory subsystem."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="MEMORY_ERROR", **kwargs)


class MemoryStoreError(MemoryError):
    """Failed to read/write to memory backend."""

    def __init__(self, backend: str, operation: str, detail: str = ""):
        super().__init__(
            f"Memory store error: {backend} {operation} failed" + (f": {detail}" if detail else ""),
            code="MEMORY_STORE_ERROR",
            detail={"backend": backend, "operation": operation, "error": detail},
        )


# --- A2A Errors ---


class A2AError(RoboClawError):
    """Agent-to-Agent protocol error."""

    def __init__(self, message: str = "", **kwargs):
        super().__init__(message, code="A2A_ERROR", **kwargs)


class AgentUnavailableError(A2AError):
    """Peer agent is not reachable."""

    def __init__(self, agent_name: str):
        super().__init__(
            f"Agent '{agent_name}' is not available",
            code="AGENT_UNAVAILABLE",
            detail={"agent_name": agent_name},
        )
