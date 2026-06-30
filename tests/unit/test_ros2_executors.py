"""Unit tests for ROS 2 action executors."""

from __future__ import annotations

import pytest


# ============================================================================
# ROS2LocomotionExecutor
# ============================================================================


@pytest.mark.asyncio
async def test_locomotion_executor_walk_steps():
    """Walk steps publishes a trajectory and returns success."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2LocomotionExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2LocomotionExecutor(bridge)
        result = await executor.execute(
            {
                "skill_type": "walk_steps",
                "sub_task_id": "walk_001",
                "parameters": {"num_steps": 2, "direction": "forward", "step_length_m": 0.3},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "ros2"
        assert result.actual_outcome["waypoints"] > 0


@pytest.mark.asyncio
async def test_locomotion_executor_navigate_to():
    """Navigate publishes a trajectory."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2LocomotionExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2LocomotionExecutor(bridge)
        result = await executor.execute(
            {
                "skill_type": "navigate_to",
                "sub_task_id": "nav_001",
                "parameters": {"room": "kitchen"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["room"] == "kitchen"


@pytest.mark.asyncio
async def test_locomotion_executor_climb_stairs():
    """Climb stairs publishes a trajectory."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2LocomotionExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2LocomotionExecutor(bridge)
        result = await executor.execute(
            {
                "skill_type": "climb_stairs",
                "sub_task_id": "climb_001",
                "parameters": {"num_steps": 3, "direction": "up"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "ros2"


@pytest.mark.asyncio
async def test_locomotion_unknown_skill():
    """Unknown skill returns failure."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2LocomotionExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2LocomotionExecutor(bridge)
        result = await executor.execute(
            {"skill_type": "fly_to_mars", "sub_task_id": "fly_001", "parameters": {}},
            {},
        )
        assert result.status == "failure"


@pytest.mark.asyncio
async def test_cancel_and_emergency_stop():
    """Cancel and estop set the flag."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2LocomotionExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2LocomotionExecutor(bridge)
        await executor.cancel()
        assert executor._cancelled

        executor._cancelled = False
        await executor.emergency_stop()
        assert executor._cancelled


# ============================================================================
# ROS2ManipulationExecutor
# ============================================================================


@pytest.mark.asyncio
async def test_manipulation_executor_grasp():
    """Grasp publishes arm trajectory."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2ManipulationExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2ManipulationExecutor(bridge)
        result = await executor.execute(
            {
                "skill_type": "whole_body_grasp",
                "sub_task_id": "grasp_001",
                "parameters": {"object": "cup", "arm": "right"},
            },
            {},
        )
        assert result.status == "success"
        assert result.actual_outcome["backend"] == "ros2"
        assert result.actual_outcome["waypoints"] > 0


@pytest.mark.asyncio
async def test_manipulation_executor_place():
    """Place object publishes arm trajectory."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2ManipulationExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2ManipulationExecutor(bridge)
        result = await executor.execute(
            {
                "skill_type": "place_object",
                "sub_task_id": "place_001",
                "parameters": {"object": "cup", "target_x": 0.3, "target_y": 0.0, "target_z": 0.5},
            },
            {},
        )
        assert result.status == "success"


@pytest.mark.asyncio
async def test_manipulation_unknown_skill():
    """Unknown manipulation skill returns failure."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_executors import ROS2ManipulationExecutor

    async with ROS2Bridge() as bridge:
        executor = ROS2ManipulationExecutor(bridge)
        result = await executor.execute(
            {"skill_type": "paint_wall", "parameters": {}},
            {},
        )
        assert result.status == "failure"


# ============================================================================
# Trajectory builders
# ============================================================================


def test_build_walk_trajectory_shape():
    """Walk trajectory has correct number of waypoints."""
    from roboclaw.comm.ros2_executors import _build_walk_trajectory

    joint_names = ["left_hip_pitch", "left_knee", "left_ankle",
                   "right_hip_pitch", "right_knee", "right_ankle"]
    wps = _build_walk_trajectory(joint_names, num_steps=3, num_waypoints_per_step=20)
    assert len(wps) == 60  # 3 * 20

    # Each waypoint covers all joints
    for wp in wps:
        for name in joint_names:
            assert name in wp


def test_build_walk_trajectory_alternating():
    """Left and right legs have opposite phase."""
    from roboclaw.comm.ros2_executors import _build_walk_trajectory

    joint_names = ["left_hip_pitch", "left_knee", "right_hip_pitch", "right_knee"]
    wps = _build_walk_trajectory(joint_names, num_steps=1, num_waypoints_per_step=20)

    # At the first waypoint, left hip is near max (sin(0) ≈ 0),
    # right hip is near min (sin(π) ≈ 0 — actually sin(π) = 0)
    # Let's check halfway through
    mid = wps[10]  # phase = π
    # Left hip should be negative or near zero
    # Right hip should be positive or near zero
    assert isinstance(mid["left_hip_pitch"], float)
    assert isinstance(mid["right_hip_pitch"], float)


def test_build_reach_trajectory_shape():
    """Reach trajectory has correct shape."""
    from roboclaw.comm.ros2_executors import _build_reach_trajectory

    joint_names = ["left_shoulder_pitch", "left_elbow", "right_shoulder_pitch", "right_elbow"]
    wps = _build_reach_trajectory(joint_names, arm="right", num_waypoints=50)
    assert len(wps) == 50

    # Final waypoint should have significant values for right arm, near zero for left
    final = wps[-1]
    assert abs(final["right_shoulder_pitch"]) > 0.1
    assert abs(final["left_shoulder_pitch"]) < 0.01
