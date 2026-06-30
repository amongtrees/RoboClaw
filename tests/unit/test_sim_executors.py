"""Unit tests for MuJoCo-based action executors."""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================================
# MujocoLocomotionExecutor
# ============================================================================


@pytest.mark.asyncio
async def test_locomotion_executor_walk_steps(mujoco_env, balance):
    """Walking a few steps produces a successful ActionResult with trajectory data."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "walk_steps",
                "sub_task_id": "walk_001",
                "parameters": {"num_steps": 3, "direction": "forward", "step_length_m": 0.3},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "mujoco"
        assert result.actual_outcome["skill"] == "walk_steps"
        assert result.actual_outcome["control_steps"] > 0
        assert result.duration_sec > 0
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_locomotion_executor_navigate_to(mujoco_env, balance):
    """Navigate-to produces a trajectory."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "navigate_to",
                "sub_task_id": "nav_001",
                "parameters": {"room": "kitchen"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "mujoco"
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_locomotion_executor_climb_stairs(mujoco_env, balance):
    """Climb stairs produces a trajectory."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "climb_stairs",
                "sub_task_id": "climb_001",
                "parameters": {"num_steps": 3, "direction": "up"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "mujoco"
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_unknown_skill_returns_failure(mujoco_env, balance):
    """An unknown locomotion skill produces a failure."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    result = await executor.execute(
        {
            "skill_type": "fly_to_moon",
            "sub_task_id": "fly_001",
            "parameters": {},
        },
        {},
    )
    assert result.status == "failure"
    assert result.error is not None
    assert "unknown_skill" in result.error.get("type", "")


@pytest.mark.asyncio
async def test_cancel_stops_execution(mujoco_env, balance):
    """Cancelling mid-execution produces an INTERRUPTED result."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    # Build a long trajectory (plan_steps returns (traj, mask) tuple)
    traj, mask = executor._controller.plan_steps(100, "forward", 0.3)
    # Cancel immediately before executing
    executor._cancelled[0] = True
    result = await executor._run_trajectory(traj, mask, "test_cancel", "walk_steps", {})
    assert result.status == "interrupted"


@pytest.mark.asyncio
async def test_emergency_stop_sets_cancelled(mujoco_env, balance):
    """Emergency stop sets the cancelled flag."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    await executor.emergency_stop()
    assert executor._cancelled[0]


# ============================================================================
# MujocoManipulationExecutor
# ============================================================================


@pytest.mark.asyncio
async def test_manipulation_executor_grasp(mujoco_env, balance):
    """Grasping produces a successful ActionResult."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoManipulationExecutor

    executor = MujocoManipulationExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "whole_body_grasp",
                "sub_task_id": "grasp_001",
                "parameters": {"object": "cup", "arm": "right"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "mujoco"
        assert result.actual_outcome["skill"] == "whole_body_grasp"
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_manipulation_executor_place(mujoco_env, balance):
    """Placing an object produces a successful ActionResult."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoManipulationExecutor

    executor = MujocoManipulationExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "place_object",
                "sub_task_id": "place_001",
                "parameters": {"object": "cup", "target_x": 0.3, "target_y": 0.0, "target_z": 0.5},
            },
            {},
        )
        assert result.status == "success"
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_manipulation_executor_handover(mujoco_env, balance):
    """Handover produces a successful ActionResult."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoManipulationExecutor

    executor = MujocoManipulationExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "handover",
                "sub_task_id": "handover_001",
                "parameters": {"object": "screwdriver", "receiver": "human"},
            },
            {},
        )
        assert result.status == "success"
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_manipulation_executor_open_door(mujoco_env, balance):
    """Open door produces a successful ActionResult."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoManipulationExecutor

    executor = MujocoManipulationExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "open_door",
                "sub_task_id": "door_001",
                "parameters": {"door": "front_door"},
            },
            {},
        )
        assert result.status == "success"
    finally:
        await executor.cancel()


@pytest.mark.asyncio
async def test_manipulation_unknown_skill(mujoco_env, balance):
    """Unknown skill returns failure."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoManipulationExecutor

    executor = MujocoManipulationExecutor(mujoco_env, balance)
    result = await executor.execute(
        {"skill_type": "juggle", "parameters": {}},
        {},
    )
    assert result.status == "failure"


# ============================================================================
# MujocoHeadNeckExecutor
# ============================================================================


@pytest.mark.asyncio
async def test_head_executor_gaze(mujoco_env, balance):
    """Gaze execution succeeds (even without neck joints)."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoHeadNeckExecutor

    executor = MujocoHeadNeckExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "gaze_at",
                "sub_task_id": "gaze_001",
                "parameters": {"target": "door"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "mujoco"
    finally:
        await executor.cancel()


# ============================================================================
# ActionResult shape
# ============================================================================


@pytest.mark.asyncio
async def test_action_result_contains_trajectory_data(mujoco_env, balance):
    """ActionResult from a MuJoCo executor includes physics trajectory data."""
    pytest.importorskip("mujoco")
    from roboclaw.sim.executors import MujocoLocomotionExecutor

    executor = MujocoLocomotionExecutor(mujoco_env, balance)
    try:
        result = await executor.execute(
            {
                "skill_type": "walk_steps",
                "sub_task_id": "walk_traj",
                "parameters": {"num_steps": 2},
            },
            {},
        )
        outcome = result.actual_outcome
        assert "control_steps" in outcome
        assert "sim_time" in outcome
        assert "final_com" in outcome
        assert "final_zmp" in outcome
        assert outcome["backend"] == "mujoco"
    finally:
        await executor.cancel()
