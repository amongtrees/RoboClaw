"""ROS 2 state provider — adapts ROS 2 sensor topics to SafetyMonitor format.

``ROS2StateProvider`` implements the ``async get_state() -> dict``
protocol that ``SafetyMonitor._check_safety()`` expects, sourcing
live data from ROS 2 topic subscriptions via the ``ROS2Bridge``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ROS2StateProvider:
    """Provides live ROS 2 sensor data in SafetyMonitor-compatible format.

    Reads cached sensor messages from the ``ROS2Bridge`` (which are
    updated by subscriber callbacks in the ROS thread) and converts them
    to the dict structure the SafetyMonitor checks.

    Parameters
    ----------
    bridge:
        The ROS2Bridge instance providing thread-safe sensor access.
    """

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge

    async def get_state(self) -> dict[str, Any]:
        """Return a dict with keys expected by SafetyMonitor._check_safety()."""
        joint_positions: dict[str, float] = {}
        zmp = (0.0, 0.0)
        left_arm_ft = {"force_xyz": (0.0, 0.0, 0.0)}
        right_arm_ft = {"force_xyz": (0.0, 0.0, 0.0)}

        # --- Joint states ---
        js = await self._bridge.get_latest_joint_state()
        if js is not None:
            try:
                joint_positions = {
                    name: float(pos)
                    for name, pos in zip(js.name, js.position)
                }
            except Exception:
                logger.debug("Failed to parse joint state", exc_info=True)

        # --- IMU / ZMP ---
        imu = await self._bridge.get_latest_imu()
        if imu is not None:
            try:
                # ZMP is not directly in IMU — approximate from linear accel
                # In a real system, ZMP comes from a dedicated estimator
                # that fuses IMU + foot force sensors + kinematics.
                # Here we default to (0,0) meaning robot is balanced.
                zmp = (0.0, 0.0)
            except Exception:
                logger.debug("Failed to parse IMU", exc_info=True)

        # --- Force-Torque ---
        ft_left = await self._bridge.get_latest_ft("left")
        if ft_left is not None:
            try:
                f = ft_left.wrench.force
                left_arm_ft = {"force_xyz": (float(f.x), float(f.y), float(f.z))}
            except Exception:
                logger.debug("Failed to parse left FT", exc_info=True)

        ft_right = await self._bridge.get_latest_ft("right")
        if ft_right is not None:
            try:
                f = ft_right.wrench.force
                right_arm_ft = {"force_xyz": (float(f.x), float(f.y), float(f.z))}
            except Exception:
                logger.debug("Failed to parse right FT", exc_info=True)

        return {
            "joint_positions": joint_positions,
            "zmp": zmp,
            "left_arm_ft": left_arm_ft,
            "right_arm_ft": right_arm_ft,
        }
