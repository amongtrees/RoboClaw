"""Integration tests for MuJoCo simulation wired into RoboClaw.

These tests verify that the full integration path works:
MujocoEnvironment → Executors → SkillLibrary → SafetyMonitor.
"""

from __future__ import annotations

import pytest


# ============================================================================
# SkillLibrary + SimModule wiring
# ============================================================================


@pytest.mark.asyncio
async def test_skill_library_wired_with_mujoco_executors(mujoco_env):
    """SkillLibrary with MuJoCo executors executes a skill with physics."""
    pytest.importorskip("mujoco")
    from roboclaw.action.skill_library import SkillLibrary
    from roboclaw.sim.module import SimModule

    lib = SkillLibrary()
    sim = SimModule.create_for_testing(mujoco_env)
    sim.wire_executors(lib)

    result = await lib.execute_skill("walk_steps", {"num_steps": 2, "direction": "forward"})
    assert result.status == "success"
    assert result.actual_outcome.get("backend") == "mujoco"
    assert result.actual_outcome.get("control_steps", 0) > 0


@pytest.mark.asyncio
async def test_skill_library_falls_back_to_simulated_without_executor(mujoco_env):
    """Without executors wired, execute_skill returns simulated fallback."""
    pytest.importorskip("mujoco")
    from roboclaw.action.skill_library import SkillLibrary

    lib = SkillLibrary()
    # Do NOT wire executors
    result = await lib.execute_skill("walk_steps", {"num_steps": 2})
    assert result.status == "success"
    assert result.actual_outcome.get("simulated") is True
    # No mujoco backend since executor wasn't registered
    assert result.actual_outcome.get("backend") != "mujoco"


# ============================================================================
# SafetyMonitor + MujocoStateProvider integration
# ============================================================================


@pytest.mark.asyncio
async def test_safety_monitor_with_mujoco_state_provider(mujoco_env):
    """SafetyMonitor with MuJoCo state provider receives live data."""
    pytest.importorskip("mujoco")
    from roboclaw.recovery.safety_monitor import SafetyMonitor
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    await mujoco_env.step()  # prime the environment

    monitor = SafetyMonitor("test_robot")
    provider = MujocoStateProvider(mujoco_env)
    monitor.set_state_provider(provider)

    # Run a single safety check (not the full loop)
    await monitor._check_safety()

    # In the nominal standing pose, there should be no violations
    assert monitor.is_safe


@pytest.mark.asyncio
async def test_safety_monitor_detects_zmp_violation(mujoco_env):
    """A ZMP outside the support polygon is flagged."""
    pytest.importorskip("mujoco")
    from roboclaw.recovery.safety_monitor import SafetyMonitor
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    await mujoco_env.step()

    class BadZmpProvider:
        async def get_state(self):
            return {
                "joint_positions": {},
                "zmp": (10.0, 10.0),  # way outside
                "left_arm_ft": {"force_xyz": (0.0, 0.0, 0.0)},
                "right_arm_ft": {"force_xyz": (0.0, 0.0, 0.0)},
                "com_position": [0.0, 0.0, 1.0],
            }

    monitor = SafetyMonitor("test_robot")
    monitor.set_state_provider(BadZmpProvider())
    await monitor._check_safety()

    violations = monitor.get_recent_violations()
    assert len(violations) > 0
    assert any("zmp" in v.violation_type for v in violations)


# ============================================================================
# Multiple skills sequential execution
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_skills_sequentially(mujoco_env):
    """Walk then grasp, verify state consistency."""
    pytest.importorskip("mujoco")
    from roboclaw.action.skill_library import SkillLibrary
    from roboclaw.sim.module import SimModule

    lib = SkillLibrary()
    sim = SimModule.create_for_testing(mujoco_env)
    sim.wire_executors(lib)

    result1 = await lib.execute_skill("walk_steps", {"num_steps": 2})
    assert result1.status == "success"

    snapshot_after_walk = mujoco_env.get_snapshot()
    assert snapshot_after_walk["step_count"] > 0

    result2 = await lib.execute_skill(
        "whole_body_grasp", {"object": "cup", "arm": "right"}
    )
    assert result2.status == "success"


# ============================================================================
# SimModule factory
# ============================================================================


@pytest.mark.asyncio
async def test_sim_module_create_for_testing(mujoco_env):
    """SimModule.create_for_testing produces a functional module."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.module import SimModule

    sim = SimModule.create_for_testing(mujoco_env)
    assert sim.env is mujoco_env
    assert sim.sim_config.enabled


def test_sim_module_create_from_config_disabled(robot_config_dict):
    """create_from_config returns None when sim is disabled."""
    from roboclaw.sim.module import SimModule

    class FakeSettings:
        robot = type('Robot', (), {
            'sim': type('Sim', (), {'enabled': False})(),
        })()

    result = SimModule.create_from_config(FakeSettings())
    assert result is None


# ============================================================================
# Config parsing
# ============================================================================


def test_sim_config_parsed_from_yaml():
    """RobotConfig.from_yaml includes sim settings when present."""
    import tempfile
    import os

    yaml_content = """
robot:
  model: "test_bot"
sim:
  enabled: true
  backend: "mujoco"
  model_path: "/tmp/test.xml"
  timestep: 0.005
  control_decimation: 2
embodiment:
  num_arms: 2
safety:
  zmp_support_polygon: []
skills:
  walk_max_speed_ms: 1.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        from roboclaw.core.config import RobotConfig
        cfg = RobotConfig.from_yaml(tmp_path)
        sim = cfg.sim
        assert sim.enabled
        assert sim.backend == "mujoco"
        assert sim.timestep == 0.005
        assert sim.control_decimation == 2
    finally:
        os.unlink(tmp_path)
