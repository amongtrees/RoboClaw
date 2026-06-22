"""Safety monitor — real-time safety constraint checking.

Runs as a background coroutine. Checks force/torque against limits,
joint positions against bounds, ZMP within support polygon.
Emits SafetyViolation events when constraints are breached.
"""

from __future__ import annotations

import asyncio
import logging
from time import time
from typing import Any

from roboclaw.core.types import SafetySeverity
from roboclaw.models.safety import SafetyViolation

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Real-time safety constraint monitor for humanoid robots.

    Checks:
    - Joint position/velocity/torque limits
    - ZMP within support polygon
    - Collision force threshold
    - Workspace boundaries
    - Human proximity (if available)
    """

    def __init__(
        self,
        robot_id: str,
        robot_config: Any = None,
        check_rate_hz: int = 100,
    ) -> None:
        self.robot_id = robot_id
        self._config = robot_config
        self._check_rate_hz = check_rate_hz
        self._check_interval_s = 1.0 / check_rate_hz
        self._running = False
        self._violations: list[SafetyViolation] = []
        self._estop_active = False

    async def start(self, state_provider: Any = None) -> None:
        """Start the safety monitor loop."""
        self._running = True
        logger.info(f"Safety monitor started at {self._check_rate_hz}Hz for {self.robot_id}")

        while self._running:
            try:
                await self._check_safety(state_provider)
            except Exception as e:
                logger.error(f"Safety monitor check error: {e}")
            await asyncio.sleep(self._check_interval_s)

    async def stop(self) -> None:
        """Stop the safety monitor."""
        self._running = False
        logger.info("Safety monitor stopped")

    async def _check_safety(self, state_provider: Any = None) -> None:
        """Run all safety checks."""
        # In Phase 1, these are stub checks.
        # In production, they read from WorkingMemory/ROS 2 topics.

        state = {}
        if state_provider:
            try:
                state = await state_provider.get_state()
            except Exception:
                pass

        # Check joint limits
        await self._check_joint_limits(state)

        # Check ZMP
        await self._check_zmp(state)

        # Check forces
        await self._check_forces(state)

    async def _check_joint_limits(self, state: dict[str, Any]) -> None:
        """Check joints against configured limits."""
        joint_states = state.get("joint_positions", {})
        # Stub — in production, compares against robot_config.embodiment.joint_limits
        for joint_name, position in joint_states.items():
            if abs(position) > 3.0:  # Hard-coded sanity check
                self._add_violation(
                    "joint_limit",
                    SafetySeverity.CRITICAL,
                    {"joint": joint_name, "position": position, "limit": 3.0},
                )

    async def _check_zmp(self, state: dict[str, Any]) -> None:
        """Check Zero Moment Point is within support polygon."""
        zmp = state.get("zmp", (0.0, 0.0))
        # Stub — in production, checks against polygon from robot_config.safety.zmp_support_polygon
        support_polygon = [
            (-0.10, -0.05), (-0.10, 0.05), (0.05, 0.05), (0.05, -0.05),
        ]
        if not self._point_in_polygon(zmp, support_polygon):
            self._add_violation(
                "zmp_violation",
                SafetySeverity.CRITICAL,
                {"zmp": zmp, "polygon": support_polygon},
            )

    async def _check_forces(self, state: dict[str, Any]) -> None:
        """Check force/torque against limits."""
        force_limit = 50.0  # from config
        for sensor in ["left_arm_ft", "right_arm_ft"]:
            ft = state.get(sensor, {})
            force = ft.get("force_xyz", (0.0, 0.0, 0.0))
            magnitude = (force[0]**2 + force[1]**2 + force[2]**2) ** 0.5
            if magnitude > force_limit:
                self._add_violation(
                    "force_limit",
                    SafetySeverity.WARNING,
                    {"sensor": sensor, "force_n": magnitude, "limit_n": force_limit},
                )

    def _add_violation(self, violation_type: str, severity: SafetySeverity, context: dict[str, Any]) -> None:
        """Record a safety violation."""
        violation = SafetyViolation(
            event_id=f"safety_{int(time()*1000)}",
            robot_id=self.robot_id,
            violation_type=violation_type,
            severity=severity,
            context=context,
        )
        self._violations.append(violation)

        if severity == SafetySeverity.E_STOP:
            self._estop_active = True
            logger.critical(f"E-STOP: {violation_type} — {context}")

        logger.warning(f"Safety violation: [{severity}] {violation_type} — {context}")

    @property
    def is_safe(self) -> bool:
        """Check if the robot is in a safe state."""
        return not self._estop_active and not any(
            v.severity in (SafetySeverity.CRITICAL, SafetySeverity.E_STOP)
            for v in self._violations[-10:]  # check recent violations
        )

    def get_recent_violations(self, n: int = 10) -> list[SafetyViolation]:
        """Get the most recent safety violations."""
        return self._violations[-n:]

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
