"""Unit tests for ROS2StateProvider."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_state_provider_returns_required_keys():
    """State dict has all keys expected by SafetyMonitor."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_state_provider import ROS2StateProvider

    async with ROS2Bridge() as bridge:
        provider = ROS2StateProvider(bridge)
        state = await provider.get_state()

        required = {"joint_positions", "zmp", "left_arm_ft", "right_arm_ft"}
        assert required <= set(state.keys())


@pytest.mark.asyncio
async def test_state_provider_stub_returns_defaults():
    """In stub mode (no sensors), returns safe defaults."""
    from roboclaw.comm.ros2_bridge import ROS2Bridge
    from roboclaw.comm.ros2_state_provider import ROS2StateProvider

    async with ROS2Bridge() as bridge:
        provider = ROS2StateProvider(bridge)
        state = await provider.get_state()

        # When no sensor data, returns empty joints and zero ZMP
        assert state["joint_positions"] == {}
        assert state["zmp"] == (0.0, 0.0)
        assert state["left_arm_ft"]["force_xyz"] == (0.0, 0.0, 0.0)
        assert state["right_arm_ft"]["force_xyz"] == (0.0, 0.0, 0.0)
