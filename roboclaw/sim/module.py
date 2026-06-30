"""Simulation module — bootstrap and wiring hub.

``SimModule`` is the single entry point for enabling MuJoCo physics
simulation.  It creates the environment, instantiates executors,
registers them with ``SkillLibrary``, and bridges sensor data to
the ``SafetyMonitor``.

Usage (in application startup)::

    from roboclaw.core.config import Settings
    from roboclaw.sim.module import SimModule

    settings = Settings.from_yaml("configs/default.yaml", "configs/robots/h1_unitree.yaml")
    sim = SimModule.create_from_config(settings)
    if sim:
        sim.wire_executors(skill_library)
        sim.wire_safety(safety_monitor)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from roboclaw.sim.balance import BalanceController
from roboclaw.sim.config import MujocoConfig, SimConfig
from roboclaw.sim.controllers import WalkingController, GraspingController, HeadController
from roboclaw.sim.executors import (
    MujocoLocomotionExecutor,
    MujocoManipulationExecutor,
    MujocoHeadNeckExecutor,
)
from roboclaw.sim.mujoco_env import MujocoEnvironment
from roboclaw.sim.mujoco_ros_bridge import MujocoROSPublisher
from roboclaw.sim.rl_walking import RLWalkingPolicy
from roboclaw.sim.sensor_bridge import MujocoStateProvider
from roboclaw.sim.speech_executor import MujocoSpeechExecutor

logger = logging.getLogger(__name__)


class SimModule:
    """Creates and wires all MuJoCo simulation components.

    Parameters
    ----------
    env:
        The shared MuJoCo physics environment.
    balance:
        The shared BalanceController for torque control + elastic band.
    config:
        The robot configuration (``RobotConfig``).
    sim_config:
        The simulation configuration (``SimConfig``).
    """

    def __init__(
        self,
        env: MujocoEnvironment,
        balance: BalanceController,
        config: Any,
        sim_config: SimConfig,
    ) -> None:
        self._env = env
        self._balance = balance
        self._config = config
        self._sim_config = sim_config

        # Lazy-initialized
        self._locomotion: MujocoLocomotionExecutor | None = None
        self._manipulation: MujocoManipulationExecutor | None = None
        self._head: MujocoHeadNeckExecutor | None = None
        self._speech: MujocoSpeechExecutor | None = None
        self._state_provider: MujocoStateProvider | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_from_config(settings: Any) -> "SimModule | None":
        """Create a SimModule from application Settings, or return None.

        Returns None when:
        - ``sim.enabled`` is False
        - MuJoCo is not installed
        - The model file is not found
        """
        robot = getattr(settings, 'robot', None)
        if robot is None:
            return None

        sim_cfg = getattr(robot, 'sim', None)
        if sim_cfg is None or isinstance(sim_cfg, dict):
            if isinstance(sim_cfg, dict) and sim_cfg.get("enabled"):
                try:
                    sim_cfg = SimConfig(**sim_cfg)
                except Exception:
                    logger.warning("Invalid sim config dict, simulation disabled")
                    return None
            else:
                return None

        if not isinstance(sim_cfg, SimConfig) or not sim_cfg.enabled:
            return None

        if not MujocoEnvironment.available():
            logger.warning(
                "sim.enabled=true but MuJoCo is not installed. "
                "Run: pip install roboclaw[sim]"
            )
            return None

        model_path = sim_cfg.model_path or getattr(robot, 'urdf_path', '')
        if not model_path or not Path(model_path).exists():
            logger.warning(
                "Sim enabled but model_path '%s' not found. "
                "Run: bash scripts/download_models.sh",
                model_path,
            )
            return None

        # Build MuJoCo config from SimConfig defaults
        mujoco_cfg = MujocoConfig()

        env = MujocoEnvironment(
            model_path,
            config=mujoco_cfg,
            control_decimation=sim_cfg.control_decimation,
        )

        # Create the shared BalanceController
        balance = BalanceController(env)

        logger.info(
            "SimModule created: model=%s, timestep=%.4f, control_rate=%.0fHz",
            model_path,
            sim_cfg.timestep,
            sim_cfg.control_rate_hz,
        )
        return SimModule(env, balance, robot, sim_cfg)

    @staticmethod
    def create_for_testing(
        env: MujocoEnvironment,
        balance: BalanceController | None = None,
        robot_config: Any = None,
    ) -> "SimModule":
        """Create a SimModule with a pre-built environment (for tests)."""
        if balance is None:
            balance = BalanceController(env)
        sim_cfg = SimConfig(enabled=True, backend="mujoco")
        return SimModule(env, balance, robot_config, sim_cfg)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    @staticmethod
    def _try_load_rl_policy() -> RLWalkingPolicy | None:
        """Try to load Unitree's pre-trained H1 walking policy.

        Returns None if PyTorch is not installed or the model file
        is not found.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.info("PyTorch not available — RL walking disabled")
            return None

        # Search for the policy in standard locations
        candidates = [
            "models/unitree_h1_walking_policy.pt",
            "models/unitree_rl_gym/deploy/pre_train/h1/motion.pt",
            "/tmp/unitree_rl_gym_check/deploy/pre_train/h1/motion.pt",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    policy = RLWalkingPolicy(path)
                    logger.info("RL walking policy loaded: %s", path)
                    return policy
                except Exception as exc:
                    logger.warning(
                        "Failed to load RL policy from %s: %s", path, exc
                    )
                    return None

        logger.info(
            "RL walking policy not found (searched: %s) — "
            "using CPG fallback. To enable, clone unitree_rl_gym: "
            "git clone https://github.com/unitreerobotics/unitree_rl_gym.git "
            "/tmp/unitree_rl_gym_check --depth 1",
            ", ".join(candidates),
        )
        return None

    def wire_executors(self, skill_library: Any, ros2_bridge: Any = None) -> None:
        """Register MuJoCo executors for all 9 skill types on the SkillLibrary.

        Parameters
        ----------
        skill_library:
            A ``SkillLibrary`` instance.
        ros2_bridge:
            Optional ``ROS2Bridge`` for ROS 2 state publishing.
        """
        # Create ROS publisher if a live bridge is available
        ros_pub = None
        if ros2_bridge is not None and ros2_bridge.is_initialized:
            ros_pub = MujocoROSPublisher(ros2_bridge, self._env.actuator_names)
            logger.info("MujocoROSPublisher attached — sim state → ROS 2 topics")

        # --- Locomotion executors ---
        if self._locomotion is None:
            # Try to load Unitree's pre-trained RL walking policy
            rl_policy = self._try_load_rl_policy()

            self._locomotion = MujocoLocomotionExecutor(
                self._env,
                self._balance,
                self._config,
                ros_publisher=ros_pub,
                controller=WalkingController(
                    joint_names=self._env.actuator_names,
                    standing_qpos=self._balance.standing_q_des,
                    step_height=getattr(
                        getattr(self._config, 'skills', None), 'walk_step_height_m', 0.05
                    ) if self._config else 0.05,
                    max_speed=getattr(
                        getattr(self._config, 'skills', None), 'walk_max_speed_ms', 1.5
                    ) if self._config else 1.5,
                ),
                rl_policy=rl_policy,
            )

        # --- Manipulation executors ---
        if self._manipulation is None:
            self._manipulation = MujocoManipulationExecutor(
                self._env,
                self._balance,
                self._config,
                ros_publisher=ros_pub,
                controller=GraspingController(
                    joint_names=self._env.actuator_names,
                    standing_qpos=self._balance.standing_q_des,
                    arm_workspace_radius=getattr(
                        getattr(self._config, 'skills', None), 'arm_workspace_radius_m', 0.85
                    ) if self._config else 0.85,
                ),
            )

        # --- Head / gaze executor ---
        if self._head is None:
            self._head = MujocoHeadNeckExecutor(
                self._env,
                self._balance,
                self._config,
                controller=HeadController(
                    joint_names=self._env.actuator_names,
                    standing_qpos=self._balance.standing_q_des,
                ),
            )

        # --- Speech executor (no physics) ---
        if self._speech is None:
            self._speech = MujocoSpeechExecutor()

        # --- Register all 9 skills ---
        # Locomotion skills (3)
        for st in ("navigate_to", "walk_steps", "climb_stairs"):
            skill_library.register_executor(st, self._locomotion)

        # Manipulation skills (4)
        for st in ("whole_body_grasp", "place_object", "handover", "open_door"):
            skill_library.register_executor(st, self._manipulation)

        # Head / gaze (1)
        skill_library.register_executor("gaze_at", self._head)

        # Speech (1)
        skill_library.register_executor("speak", self._speech)

        logger.info(
            "SimModule wired %d skill executors to SkillLibrary (9 skills total)",
            9,
        )

    def wire_safety(self, safety_monitor: Any) -> None:
        """Set the MuJoCo state provider on the SafetyMonitor."""
        if self._state_provider is None:
            self._state_provider = MujocoStateProvider(self._env)

        safety_monitor.set_state_provider(self._state_provider)
        logger.info("SimModule wired MujocoStateProvider to SafetyMonitor")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Reset the simulation environment and re-settle balance."""
        await self._env.reset()
        self._balance.settle(self._env, steps=200)
        logger.info("SimModule reset")

    async def close(self) -> None:
        """Release simulation resources."""
        self._env.close()
        logger.info("SimModule closed")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def env(self) -> MujocoEnvironment:
        """The shared MuJoCo environment."""
        return self._env

    @property
    def balance(self) -> BalanceController:
        """The shared BalanceController."""
        return self._balance

    @property
    def config(self) -> Any:
        """The robot configuration."""
        return self._config

    @property
    def sim_config(self) -> SimConfig:
        """The simulation configuration."""
        return self._sim_config
