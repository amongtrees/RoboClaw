"""Shared test fixtures for RoboClaw tests."""

from __future__ import annotations

import pytest

from roboclaw.core.registry import ComponentRegistry, reset_registry


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset the global component registry before each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def reset_sensor_fusion_singleton():
    """Reset the SensorFusion singleton between tests."""
    from roboclaw.orchestration import nodes as orch_nodes

    old = getattr(orch_nodes, '_sensor_fusion', None)
    orch_nodes._sensor_fusion = None
    yield
    orch_nodes._sensor_fusion = old


@pytest.fixture
def registry():
    """Get a fresh component registry."""
    return ComponentRegistry()


@pytest.fixture
def robot_config_dict():
    """Sample robot configuration for testing."""
    return {
        "model": "h1_unitree",
        "display_name": "Unitree H1",
        "embodiment": {
            "num_arms": 2,
            "num_legs": 2,
            "num_fingers_per_hand": 5,
            "height_m": 1.80,
            "mass_kg": 47.0,
        },
        "safety": {
            "zmp_support_polygon": [[-0.10, -0.05], [-0.10, 0.05], [0.05, 0.05], [0.05, -0.05]],
            "max_joint_velocity_rad_s": 5.0,
            "collision_force_threshold_n": 50.0,
        },
        "skills": {
            "default_grasp_strategy": "top_down_encompassing",
            "walk_max_speed_ms": 1.5,
        },
    }


@pytest.fixture
def sample_task_spec():
    """Sample task specification for testing."""
    return {
        "task_id": "task_test_001",
        "robot_id": "h1_test",
        "goal": "go to the kitchen and pick up the cup",
        "priority": 0,
    }


@pytest.fixture
def sample_robot_state():
    """Sample robot state for testing."""
    from roboclaw.models.state import HumanoidState

    return HumanoidState(
        robot_id="h1_test",
        phase="idle",
        joint_states={
            "left_shoulder_pitch": 0.0,
            "right_shoulder_pitch": 0.0,
            "left_knee": 0.0,
            "right_knee": 0.0,
        },
        zmp=(0.0, 0.0),
        support_phase="double",
    )


# ---------------------------------------------------------------------------
# MuJoCo simulation fixtures
# ---------------------------------------------------------------------------

MINIMAL_HUMANOID_XML = """
<mujoco model="test_humanoid">
  <compiler angle="radian" coordinate="local" inertiafromgeom="auto"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <geom type="capsule" rgba="0.6 0.6 0.8 1" condim="3"/>
    <joint type="hinge" damping="1.0" armature="0.01"/>
    <motor ctrlrange="-1.0 1.0" gear="50"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.3 0.3 0.3 1"/>

    <!-- Torso with free joint -->
    <body name="torso" pos="0 0 1.0">
      <joint name="root" type="free" limited="false"/>
      <geom name="torso_geom" type="box" size="0.12 0.2 0.25" mass="25" rgba="0.4 0.5 0.7 1"/>

      <!-- Left leg -->
      <body name="left_thigh" pos="0 -0.1 -0.15">
        <joint name="left_hip_pitch" axis="0 1 0" range="-1.5 0.5" limited="true"/>
        <geom name="left_thigh_geom" size="0.06 0.35" mass="5" fromto="0 0 0 0 0 -0.35"/>
        <body name="left_shin" pos="0 0 -0.35">
          <joint name="left_knee" axis="0 1 0" range="0.0 2.5" limited="true"/>
          <geom name="left_shin_geom" size="0.05 0.35" mass="3" fromto="0 0 0 0 0 -0.35"/>
          <body name="left_foot" pos="0 0 -0.35">
            <geom name="left_foot_geom" type="box" size="0.04 0.1 0.03" pos="0.03 0 0" mass="1" rgba="0.2 0.2 0.2 1"/>
          </body>
        </body>
      </body>

      <!-- Right leg -->
      <body name="right_thigh" pos="0 0.1 -0.15">
        <joint name="right_hip_pitch" axis="0 1 0" range="-1.5 0.5" limited="true"/>
        <geom name="right_thigh_geom" size="0.06 0.35" mass="5" fromto="0 0 0 0 0 -0.35"/>
        <body name="right_shin" pos="0 0 -0.35">
          <joint name="right_knee" axis="0 1 0" range="0.0 2.5" limited="true"/>
          <geom name="right_shin_geom" size="0.05 0.35" mass="3" fromto="0 0 0 0 0 -0.35"/>
          <body name="right_foot" pos="0 0 -0.35">
            <geom name="right_foot_geom" type="box" size="0.04 0.1 0.03" pos="0.03 0 0" mass="1" rgba="0.2 0.2 0.2 1"/>
          </body>
        </body>
      </body>

      <!-- Left arm -->
      <body name="left_upper_arm" pos="0 -0.25 0.2">
        <joint name="left_shoulder_pitch" axis="0 1 0" range="-2.0 2.0" limited="true"/>
        <geom name="left_upper_arm_geom" size="0.04 0.3" mass="2" fromto="0 0 0 0 0 -0.3"/>
        <body name="left_forearm" pos="0 0 -0.3">
          <joint name="left_elbow" axis="0 1 0" range="-2.5 0.0" limited="true"/>
          <geom name="left_forearm_geom" size="0.03 0.25" mass="1.5" fromto="0 0 0 0 0 -0.25"/>
          <body name="left_hand" pos="0 0 -0.25">
            <geom name="left_hand_geom" type="box" size="0.02 0.05 0.08" mass="0.5" rgba="0.9 0.8 0.7 1"/>
          </body>
        </body>
      </body>

      <!-- Right arm -->
      <body name="right_upper_arm" pos="0 0.25 0.2">
        <joint name="right_shoulder_pitch" axis="0 1 0" range="-2.0 2.0" limited="true"/>
        <geom name="right_upper_arm_geom" size="0.04 0.3" mass="2" fromto="0 0 0 0 0 -0.3"/>
        <body name="right_forearm" pos="0 0 -0.3">
          <joint name="right_elbow" axis="0 1 0" range="-2.5 0.0" limited="true"/>
          <geom name="right_forearm_geom" size="0.03 0.25" mass="1.5" fromto="0 0 0 0 0 -0.25"/>
          <body name="right_hand" pos="0 0 -0.25">
            <geom name="right_hand_geom" type="box" size="0.02 0.05 0.08" mass="0.5" rgba="0.9 0.8 0.7 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_hip_pitch_act"    joint="left_hip_pitch"    gear="100"/>
    <motor name="left_knee_act"         joint="left_knee"         gear="80"/>
    <motor name="right_hip_pitch_act"   joint="right_hip_pitch"   gear="100"/>
    <motor name="right_knee_act"        joint="right_knee"        gear="80"/>
    <motor name="left_shoulder_pitch_act" joint="left_shoulder_pitch" gear="40"/>
    <motor name="left_elbow_act"        joint="left_elbow"        gear="30"/>
    <motor name="right_shoulder_pitch_act" joint="right_shoulder_pitch" gear="40"/>
    <motor name="right_elbow_act"       joint="right_elbow"       gear="30"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def minimal_mujoco_xml() -> str:
    """Return a minimal humanoid MJCF XML for unit testing.

    This model has a torso (free joint), two legs (hip + knee),
    two arms (shoulder + elbow), ground plane, and 8 actuators.
    It is ~80 lines of XML and requires no external files.
    """
    return MINIMAL_HUMANOID_XML


@pytest.fixture
def mujoco_env(minimal_mujoco_xml: str):
    """Create a MuJoCoEnvironment from the minimal test model.

    Uses ``pytest.importorskip`` so the entire test suite passes
    when MuJoCo is not installed.
    """
    pytest.importorskip("mujoco")
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    return MujocoEnvironment.from_string(minimal_mujoco_xml)


@pytest.fixture
def balance(mujoco_env):
    """Create a BalanceController from the MuJoCo environment."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.balance import BalanceController

    return BalanceController(env=mujoco_env)


@pytest.fixture
def sim_config_dict() -> dict:
    """Sample simulation configuration for testing."""
    return {
        "enabled": True,
        "backend": "mujoco",
        "model_path": "",
        "render": False,
        "headless": True,
        "timestep": 0.005,
        "control_decimation": 2,
        "sensor_rate_hz": 100,
    }
