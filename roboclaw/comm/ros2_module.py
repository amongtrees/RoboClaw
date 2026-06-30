"""ROS 2 module — bootstrap and wiring hub.

``ROS2Module`` is the single entry point for enabling ROS 2 hardware
integration.  It creates the ``ROS2Bridge``, instantiates ROS 2
executors, registers them with ``SkillLibrary``, and bridges sensor
data to the ``SafetyMonitor``.

Follows the same pattern as ``SimModule`` for consistency.

Usage (in application startup)::

    from roboclaw.core.config import Settings
    from roboclaw.comm.ros2_module import ROS2Module

    settings = Settings.from_yaml("configs/default.yaml", "configs/robots/h1_unitree.yaml")
    ros2 = ROS2Module.create_from_config(settings)
    if ros2:
        await ros2.initialize()
        ros2.wire_executors(skill_library)
        ros2.wire_safety(safety_monitor)
"""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.comm.ros2_bridge import ROS2Bridge
from roboclaw.comm.ros2_executors import (
    ROS2LocomotionExecutor,
    ROS2ManipulationExecutor,
)
from roboclaw.comm.ros2_state_provider import ROS2StateProvider

logger = logging.getLogger(__name__)


class ROS2Module:
    """Creates and wires all ROS 2 hardware integration components.

    Parameters
    ----------
    bridge:
        The ROS2Bridge instance.
    config:
        The robot configuration (``RobotConfig``).
    """

    def __init__(
        self,
        bridge: ROS2Bridge,
        config: Any = None,
    ) -> None:
        self._bridge = bridge
        self._config = config

        # Lazy-initialized
        self._locomotion: ROS2LocomotionExecutor | None = None
        self._manipulation: ROS2ManipulationExecutor | None = None
        self._state_provider: ROS2StateProvider | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_from_config(settings: Any) -> "ROS2Module | None":
        """Create a ROS2Module from application Settings, or return None.

        Returns None when ros2 is not enabled in config.
        """
        robot = getattr(settings, 'robot', None)
        if robot is None:
            return None

        ros2_cfg = getattr(robot, 'ros2', None)
        if ros2_cfg is None:
            return None

        if isinstance(ros2_cfg, dict):
            enabled = ros2_cfg.get("enabled", False)
        elif hasattr(ros2_cfg, 'enabled'):
            enabled = ros2_cfg.enabled
        else:
            enabled = False

        if not enabled:
            logger.debug("ros2.enabled is false — ROS 2 module not created")
            return None

        node_name = "roboclaw_bridge"
        if isinstance(ros2_cfg, dict):
            node_name = ros2_cfg.get("node_name", node_name)
        elif hasattr(ros2_cfg, 'node_name'):
            node_name = ros2_cfg.node_name

        bridge = ROS2Bridge(robot_config=robot, node_name=node_name)
        logger.info("ROS2Module created: node=%s", node_name)
        return ROS2Module(bridge, robot)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start the ROS 2 bridge (background thread)."""
        await self._bridge.initialize()

    async def shutdown(self) -> None:
        """Stop the ROS 2 bridge."""
        await self._bridge.shutdown()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def wire_executors(self, skill_library: Any) -> None:
        """Register ROS 2 executors for locomotion and manipulation skills.

        Parameters
        ----------
        skill_library:
            A ``SkillLibrary`` instance.
        """
        # Read joint name mappings from config
        leg_joints = self._get_joint_names("leg")
        arm_joints = self._get_joint_names("arm")

        if self._locomotion is None:
            self._locomotion = ROS2LocomotionExecutor(
                self._bridge,
                joint_names=leg_joints,
                robot_config=self._config,
            )
        if self._manipulation is None:
            self._manipulation = ROS2ManipulationExecutor(
                self._bridge,
                arm_joint_names=arm_joints,
                robot_config=self._config,
            )

        # --- Locomotion skills ---
        for st in ("navigate_to", "walk_steps", "climb_stairs"):
            skill_library.register_executor(st, self._locomotion)

        # --- Manipulation skills ---
        for st in ("whole_body_grasp", "place_object", "handover", "open_door"):
            skill_library.register_executor(st, self._manipulation)

        logger.info(
            "ROS2Module wired %d skill executors to SkillLibrary",
            7,
        )

    def wire_safety(self, safety_monitor: Any) -> None:
        """Set the ROS 2 state provider on the SafetyMonitor.

        Parameters
        ----------
        safety_monitor:
            A ``SafetyMonitor`` instance.
        """
        if self._state_provider is None:
            self._state_provider = ROS2StateProvider(self._bridge)

        safety_monitor.set_state_provider(self._state_provider)
        logger.info("ROS2Module wired ROS2StateProvider to SafetyMonitor")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_joint_names(self, group: str) -> list[str] | None:
        """Extract joint name list from robot config if available."""
        if self._config is None:
            return None
        ros2_cfg = getattr(self._config, 'ros2', None) or {}
        if isinstance(ros2_cfg, dict):
            joint_names = ros2_cfg.get("joint_names", {})
            return joint_names.get(group, None)
        return None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def bridge(self) -> ROS2Bridge:
        """The ROS 2 bridge."""
        return self._bridge

    @property
    def is_initialized(self) -> bool:
        """True if the bridge has a running ROS 2 node."""
        return self._bridge.is_initialized
