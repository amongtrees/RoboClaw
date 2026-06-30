"""Unit tests for MujocoEnvironment."""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================================
# Availability
# ============================================================================


def test_available_returns_bool():
    """MujocoEnvironment.available() returns a boolean."""
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    result = MujocoEnvironment.available()
    assert isinstance(result, bool)


def test_available_caches_result():
    """Multiple calls to available() return the same result."""
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    a = MujocoEnvironment.available()
    b = MujocoEnvironment.available()
    assert a == b


# ============================================================================
# Environment creation
# ============================================================================


def test_from_string_creates_model(minimal_mujoco_xml: str):
    """MujocoEnvironment.from_string() creates a valid environment."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    env = MujocoEnvironment.from_string(minimal_mujoco_xml)
    try:
        assert env.n_joints > 0
        assert env.n_actuators > 0
        assert env.n_qpos > 0
        assert env.n_qvel > 0
        assert not env.closed
    finally:
        env.close()


def test_from_string_joint_names(minimal_mujoco_xml: str):
    """Joint names are extracted after construction."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    env = MujocoEnvironment.from_string(minimal_mujoco_xml)
    try:
        names = env.joint_names
        assert "root" in names
        assert "left_hip_pitch" in names
        assert "left_knee" in names
        assert "right_hip_pitch" in names
        assert "right_knee" in names
        assert "left_shoulder_pitch" in names
        assert "left_elbow" in names
        assert "right_shoulder_pitch" in names
        assert "right_elbow" in names
    finally:
        env.close()


def test_from_string_actuator_names(minimal_mujoco_xml: str):
    """Actuator names are extracted after construction."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    env = MujocoEnvironment.from_string(minimal_mujoco_xml)
    try:
        names = env.actuator_names
        assert len(names) == 8
        assert "left_hip_pitch_act" in names
    finally:
        env.close()


def test_from_file_raises_if_missing():
    """Constructor raises FileNotFoundError for missing model."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    with pytest.raises(FileNotFoundError):
        MujocoEnvironment("/nonexistent/path/model.xml")


# ============================================================================
# Physics stepping — these are async because env.step() is async
# ============================================================================


async def test_step_advances_time(mujoco_env):
    """After calling step(), sim time increases."""
    snapshot = await mujoco_env.step()
    assert snapshot["time"] > 0.0
    assert mujoco_env.time > 0.0


async def test_step_increments_step_count(mujoco_env):
    """Each step() call increments the step counter."""
    assert mujoco_env.step_count == 0
    await mujoco_env.step()
    assert mujoco_env.step_count == 1
    await mujoco_env.step()
    assert mujoco_env.step_count == 2


async def test_step_with_zero_control(mujoco_env):
    """Stepping with explicit zero control works."""
    ctrl = np.zeros(mujoco_env.n_actuators)
    snapshot = await mujoco_env.step(ctrl)
    assert snapshot is not None
    assert "joint_positions" in snapshot


async def test_step_raises_on_wrong_ctrl_shape(mujoco_env):
    """ValueError when control shape doesn't match number of actuators."""
    with pytest.raises(ValueError):
        await mujoco_env.step(np.array([1.0]))  # too short

    with pytest.raises(ValueError):
        await mujoco_env.step(np.zeros(mujoco_env.n_actuators + 10))


# ============================================================================
# Reset
# ============================================================================


async def test_reset_restores_initial_state(mujoco_env):
    """reset() restores time=0 and qpos=qpos0."""
    await mujoco_env.step()
    await mujoco_env.step()

    snapshot = await mujoco_env.reset()
    assert snapshot["time"] == 0.0
    assert mujoco_env.step_count == 0
    assert mujoco_env.time == 0.0


async def test_reset_with_custom_qpos(mujoco_env):
    """reset() with given qpos sets those positions."""
    qpos = mujoco_env.model.qpos0.copy()
    qpos[2] += 0.1  # raise torso a bit
    snapshot = await mujoco_env.reset(qpos)
    assert snapshot["time"] == 0.0


async def test_reset_raises_on_wrong_qpos_shape(mujoco_env):
    """ValueError when qpos shape doesn't match."""
    with pytest.raises(ValueError):
        await mujoco_env.reset(np.array([0.0]))  # too short


# ============================================================================
# Snapshot
# ============================================================================


async def test_snapshot_has_required_keys(mujoco_env):
    """Snapshot dict contains all expected keys."""
    await mujoco_env.step()
    snapshot = mujoco_env.get_snapshot()

    required_keys = {
        "time", "step_count", "qpos", "qvel", "joint_positions",
        "joint_velocities", "com_position", "zmp", "contact_forces",
        "total_contact_force", "imu", "body_poses",
    }
    assert required_keys <= set(snapshot.keys())


async def test_snapshot_joint_positions_named(mujoco_env):
    """joint_positions has named entries matching model joints."""
    await mujoco_env.step()
    snapshot = mujoco_env.get_snapshot()

    jp = snapshot["joint_positions"]
    for name in mujoco_env.joint_names:
        if name == "root":
            # Free joint: split into pos + quat components
            assert f"{name}_z" in jp
        else:
            assert name in jp, f"Missing joint: {name}"


async def test_snapshot_zmp_is_2d(mujoco_env):
    """ZMP is a 2-element list."""
    await mujoco_env.step()
    snapshot = mujoco_env.get_snapshot()

    zmp = snapshot["zmp"]
    assert isinstance(zmp, list)
    assert len(zmp) == 2


async def test_snapshot_com_is_3d(mujoco_env):
    """COM position is a 3-element list."""
    await mujoco_env.step()
    snapshot = mujoco_env.get_snapshot()

    com = snapshot["com_position"]
    assert isinstance(com, list)
    assert len(com) == 3


def test_get_snapshot_empty_before_step(minimal_mujoco_xml: str):
    """get_snapshot() returns empty dict before any step."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    env = MujocoEnvironment.from_string(minimal_mujoco_xml)
    try:
        snapshot = env.get_snapshot()
        assert snapshot == {}
    finally:
        env.close()


# ============================================================================
# Body poses
# ============================================================================


async def test_body_poses_contains_key_bodies(mujoco_env):
    """body_poses contains known body names after step."""
    await mujoco_env.step()
    snapshot = mujoco_env.get_snapshot()

    body_poses = snapshot["body_poses"]
    # At minimum, the torso should be present
    assert "torso" in body_poses
    assert "position" in body_poses["torso"]
    assert "rotation" in body_poses["torso"]


# ============================================================================
# Close
# ============================================================================


def test_close_marks_closed(mujoco_env):
    """After close(), closed is True."""
    mujoco_env.close()
    assert mujoco_env.closed


async def test_step_after_close_raises(mujoco_env):
    """RuntimeError when stepping a closed environment."""
    mujoco_env.close()
    with pytest.raises(RuntimeError):
        await mujoco_env.step()


# ============================================================================
# Config
# ============================================================================


def test_custom_config_gravity(minimal_mujoco_xml: str):
    """Custom gravity is applied to the model."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.config import MujocoConfig
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    cfg = MujocoConfig(gravity=[0.0, 0.0, -1.62])  # moon gravity
    env = MujocoEnvironment.from_string(minimal_mujoco_xml, config=cfg)
    try:
        g = env.model.opt.gravity
        assert g[2] == -1.62
    finally:
        env.close()


def test_custom_config_timestep(minimal_mujoco_xml: str):
    """Custom config fields are stored."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.config import MujocoConfig
    from roboclaw.sim.mujoco_env import MujocoEnvironment

    cfg = MujocoConfig(solver="CG", iterations=50, tolerance=1e-6)
    env = MujocoEnvironment.from_string(minimal_mujoco_xml, config=cfg)
    try:
        assert env._config.solver == "CG"
        assert env._config.iterations == 50
    finally:
        env.close()
