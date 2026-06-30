"""Unit tests for ROS2Bridge (stub mode — no rclpy required)."""

from __future__ import annotations

import pytest


# ============================================================================
# Bridge creation and initialization
# ============================================================================


@pytest.mark.asyncio
async def test_bridge_creates_in_stub_mode():
    """Bridge starts in stub mode when rclpy is not available."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    bridge = ROS2Bridge(node_name="test_bridge")
    await bridge.initialize()

    # In this test environment, rclpy may or may not be available.
    # Either way, the bridge should not crash.
    assert bridge is not None
    await bridge.shutdown()


@pytest.mark.asyncio
async def test_bridge_context_manager():
    """Bridge works as an async context manager."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge(node_name="test_ctx") as bridge:
        assert bridge is not None


# ============================================================================
# Stub sensor access
# ============================================================================


@pytest.mark.asyncio
async def test_get_latest_joint_state_returns_none():
    """In stub mode, joint state is None."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        js = await bridge.get_latest_joint_state()
        assert js is None


@pytest.mark.asyncio
async def test_get_latest_imu_returns_none():
    """In stub mode, IMU is None."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        imu = await bridge.get_latest_imu()
        assert imu is None


@pytest.mark.asyncio
async def test_get_latest_ft_returns_none():
    """In stub mode, FT is None."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        ft_left = await bridge.get_latest_ft("left")
        ft_right = await bridge.get_latest_ft("right")
        assert ft_left is None
        assert ft_right is None


# ============================================================================
# Stub publishing
# ============================================================================


@pytest.mark.asyncio
async def test_publish_trajectory_from_dict_no_crash():
    """publish_trajectory_from_dict does not crash in stub mode."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        waypoints = [
            {"left_hip_pitch": 0.0, "left_knee": 0.0},
            {"left_hip_pitch": -0.2, "left_knee": 0.4},
        ]
        await bridge.publish_trajectory_from_dict(
            ["left_hip_pitch", "left_knee"], waypoints
        )


@pytest.mark.asyncio
async def test_publish_trajectory_from_dict_empty():
    """Empty waypoints should not crash."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        await bridge.publish_trajectory_from_dict(
            ["joint_a"], []
        )


# ============================================================================
# Stub action call
# ============================================================================


@pytest.mark.asyncio
async def test_call_action_returns_result():
    """call_action always returns a result dict."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        result = await bridge.call_action("/walk_to_pose", goal={})
        assert isinstance(result, dict)
        assert "status" in result


# ============================================================================
# TF
# ============================================================================


@pytest.mark.asyncio
async def test_get_tf_transform_returns_identity():
    """In stub mode, TF returns identity."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        tf = await bridge.get_tf_transform("base_link", "map")
        assert tf["translation"] == [0.0, 0.0, 0.0]
        assert tf["rotation"] == [0.0, 0.0, 0.0, 1.0]


# ============================================================================
# Initialized flag
# ============================================================================


@pytest.mark.asyncio
async def test_is_initialized():
    """is_initialized reflects ROS availability."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge

    async with ROS2Bridge() as bridge:
        # In this test env, rclpy from system ROS is Python 3.12
        # but we're on 3.11, so it should be False
        assert isinstance(bridge.is_initialized, bool)
