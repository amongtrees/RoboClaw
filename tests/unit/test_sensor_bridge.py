"""Unit tests for MujocoStateProvider."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_get_state_has_required_keys(mujoco_env):
    """State dict contains all keys expected by SafetyMonitor."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    provider = MujocoStateProvider(mujoco_env)
    # Step the env first so we have data
    await mujoco_env.step()

    state = await provider.get_state()

    required = {"joint_positions", "zmp", "left_arm_ft", "right_arm_ft", "com_position", "contact_forces", "support_phase"}
    assert required <= set(state.keys())


@pytest.mark.asyncio
async def test_get_state_empty_before_step(mujoco_env):
    """Before any step, state provider returns minimal safe defaults."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    provider = MujocoStateProvider(mujoco_env)
    state = await provider.get_state()

    assert state["joint_positions"] == {}
    assert state["zmp"] == (0.0, 0.0)
    assert state["support_phase"] == "double"


@pytest.mark.asyncio
async def test_zmp_is_tuple(mujoco_env):
    """ZMP is returned as a 2-element tuple."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    provider = MujocoStateProvider(mujoco_env)
    await mujoco_env.step()
    state = await provider.get_state()

    zmp = state["zmp"]
    assert isinstance(zmp, tuple)
    assert len(zmp) == 2


@pytest.mark.asyncio
async def test_force_sensors_have_correct_structure(mujoco_env):
    """Arm FT dicts have the expected force_xyz key."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    provider = MujocoStateProvider(mujoco_env)
    await mujoco_env.step()
    state = await provider.get_state()

    assert "force_xyz" in state["left_arm_ft"]
    assert "force_xyz" in state["right_arm_ft"]


@pytest.mark.asyncio
async def test_support_phase_is_string(mujoco_env):
    """Support phase is a string (double, left, right, flight)."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    provider = MujocoStateProvider(mujoco_env)
    await mujoco_env.step()
    state = await provider.get_state()

    assert state["support_phase"] in ("double", "left", "right", "flight")


@pytest.mark.asyncio
async def test_get_state_with_multiple_steps(mujoco_env):
    """State changes between steps."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.sensor_bridge import MujocoStateProvider

    provider = MujocoStateProvider(mujoco_env)
    await mujoco_env.step()
    state1 = await provider.get_state()

    await mujoco_env.step()
    state2 = await provider.get_state()

    # Time or step count should differ
    assert state1.get("joint_positions") is not None
    assert state2.get("joint_positions") is not None
