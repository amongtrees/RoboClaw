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
