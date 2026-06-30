"""RoboClaw simulation package.

This package provides MuJoCo-based physics simulation for humanoid
robotics tasks.  When ``sim.enabled`` is true (in the robot config),
the MuJoCo executors replace the ``asyncio.sleep()`` stubs with
real physics stepping.

Core components
---------------
- ``MujocoEnvironment``: Wraps MuJoCo model/data, steps physics, extracts sensors.
- ``MujocoLocomotionExecutor``: Walking, stair climbing via MuJoCo physics.
- ``MujocoManipulationExecutor``: Grasp, place, handover via MuJoCo physics.
- ``MujocoHeadNeckExecutor``: Gaze control via MuJoCo physics.
- ``MujocoStateProvider``: Adapts MuJoCo state for the SafetyMonitor.
- ``SimModule``: Bootstrap that wires everything into SkillLibrary, SafetyMonitor.

Quick start::

    from roboclaw.sim import SimModule, mujoco_available
    if mujoco_available():
        sim = SimModule.create_from_config(settings)
        if sim:
            sim.wire_executors(skill_library)
            sim.wire_safety(safety_monitor)
"""

from roboclaw.sim.balance import BalanceController
from roboclaw.sim.config import MujocoConfig, SimConfig
from roboclaw.sim.controllers import WalkingController, GraspingController, HeadController, StandingController
from roboclaw.sim.executors import (
    MujocoLocomotionExecutor,
    MujocoManipulationExecutor,
    MujocoHeadNeckExecutor,
)
from roboclaw.sim.module import SimModule
from roboclaw.sim.mujoco_env import MujocoEnvironment
from roboclaw.sim.mujoco_ros_bridge import MujocoROSPublisher
from roboclaw.sim.sensor_bridge import MujocoStateProvider
from roboclaw.sim.speech_executor import MujocoSpeechExecutor


def mujoco_available() -> bool:
    """Return True if MuJoCo Python bindings are installed."""
    return MujocoEnvironment.available()


__all__ = [
    "BalanceController",
    "SimConfig",
    "MujocoConfig",
    "MujocoEnvironment",
    "MujocoLocomotionExecutor",
    "MujocoManipulationExecutor",
    "MujocoHeadNeckExecutor",
    "MujocoSpeechExecutor",
    "MujocoROSPublisher",
    "MujocoStateProvider",
    "SimModule",
    "WalkingController",
    "GraspingController",
    "HeadController",
    "StandingController",
    "mujoco_available",
]
