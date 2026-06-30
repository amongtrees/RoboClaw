"""Tests for roboclaw.perception.sensor_fusion."""

from __future__ import annotations

import pytest

from roboclaw.orchestration.state_schema import AgentGraphState, PerceptionData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AgentGraphState:
    """Build a minimal AgentGraphState for testing."""
    defaults = {
        "robot_id": "test_robot",
        "world_objects": [{"id": "cup", "label": "cup", "type": "object"}],
    }
    defaults.update(overrides)
    return AgentGraphState(**defaults)


# ---------------------------------------------------------------------------
# Stub fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensor_fusion_stub_without_env():
    """When no environment is attached, fuse() returns stub data."""
    from roboclaw.perception.sensor_fusion import SensorFusion

    sf = SensorFusion(env=None)
    state = _make_state()
    result = await sf.fuse(state)

    assert "perception" in result
    perception = result["perception"]
    assert isinstance(perception, PerceptionData)
    assert perception.timestamp > 0
    assert len(perception.detected_objects) == 1
    assert perception.detected_objects[0]["id"] == "cup"


@pytest.mark.asyncio
async def test_sensor_fusion_stub_preserves_world_objects():
    """Stub path should pass through world_objects as detected_objects."""
    from roboclaw.perception.sensor_fusion import SensorFusion

    sf = SensorFusion(env=None)
    state = _make_state(world_objects=[
        {"id": "obj_1", "label": "table"},
        {"id": "obj_2", "label": "door"},
    ])
    result = await sf.fuse(state)

    perception = result["perception"]
    assert len(perception.detected_objects) == 2


# ---------------------------------------------------------------------------
# Real sensor fusion with MuJoCo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensor_fusion_with_env(mujoco_env):
    """With a MuJoCo environment, fuse() returns real sensor data."""
    from roboclaw.perception.sensor_fusion import SensorFusion

    await mujoco_env.step(None)  # one step to populate snapshot (None = zero ctrl)
    sf = SensorFusion(env=mujoco_env)
    state = _make_state()
    result = await sf.fuse(state)

    # Check that sensor fields are populated
    assert "joint_positions" in result
    assert isinstance(result["joint_positions"], dict)
    assert len(result["joint_positions"]) > 0, "Should have named joint positions"

    assert "joint_velocities" in result
    assert isinstance(result["joint_velocities"], dict)

    assert "imu_data" in result
    assert "accelerometer" in result["imu_data"]
    assert "gyroscope" in result["imu_data"]

    assert "body_poses" in result
    assert isinstance(result["body_poses"], dict)
    assert "torso" in result["body_poses"], "Should contain torso body pose"

    assert "contact_forces" in result
    assert isinstance(result["contact_forces"], list)

    assert "com_position" in result
    assert isinstance(result["com_position"], list)
    assert len(result["com_position"]) == 3

    assert "zmp_position" in result
    assert isinstance(result["zmp_position"], list)
    assert len(result["zmp_position"]) == 2

    assert "proprioceptive_state" in result
    assert "joint_positions" in result["proprioceptive_state"]
    assert "imu" in result["proprioceptive_state"]

    # Perception should be populated
    perception = result["perception"]
    assert isinstance(perception, PerceptionData)
    assert perception.timestamp > 0
    # detected_objects should include both world objects and body parts
    body_object_ids = [o["id"] for o in perception.detected_objects if o.get("type") == "robot_body_part"]
    assert len(body_object_ids) > 0, "Should detect robot body parts"

    # Scene graph should be built
    assert "nodes" in perception.scene_graph
    assert len(perception.scene_graph["nodes"]) > 0

    # Robot pose should be derived from torso
    if "robot_pose" in result:
        rp = result["robot_pose"]
        if rp:
            assert "position" in rp


@pytest.mark.asyncio
async def test_sensor_fusion_empty_snapshot_before_step(mujoco_env):
    """Before stepping, an empty snapshot should fall back to stub."""
    from roboclaw.perception.sensor_fusion import SensorFusion

    sf = SensorFusion(env=mujoco_env)
    state = _make_state()
    result = await sf.fuse(state)

    # Empty snapshot → stub fallback
    perception = result["perception"]
    assert isinstance(perception, PerceptionData)
    # Should still work but may have minimal data
    assert perception.timestamp > 0


# ---------------------------------------------------------------------------
# perceive_node integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perceive_node_with_sensor_fusion(mujoco_env):
    """perceive_node should use SensorFusion when singleton is set."""
    from roboclaw.perception.sensor_fusion import SensorFusion
    from roboclaw.orchestration.nodes import perceive_node, set_sensor_fusion, _sensor_fusion as sf_global
    import roboclaw.orchestration.nodes as nodes_mod

    await mujoco_env.step(None)
    sf = SensorFusion(env=mujoco_env)
    set_sensor_fusion(sf)

    try:
        state = _make_state()
        result = await perceive_node(state)

        # Should have real sensor data from fusion
        assert "joint_positions" in result
        assert len(result["joint_positions"]) > 0
        assert result["loop_count"] == state.loop_count + 1
        assert "phase" in result
    finally:
        # Clean up singleton
        nodes_mod._sensor_fusion = None


@pytest.mark.asyncio
async def test_perceive_node_fallback_without_sensor_fusion():
    """perceive_node should use stubs when no SensorFusion singleton is set."""
    from roboclaw.orchestration.nodes import perceive_node
    import roboclaw.orchestration.nodes as nodes_mod

    # Ensure singleton is cleared
    nodes_mod._sensor_fusion = None

    state = _make_state()
    result = await perceive_node(state)

    # Should work with stub data
    assert "perception" in result
    assert isinstance(result["perception"], PerceptionData)
    assert result["loop_count"] == state.loop_count + 1


# ---------------------------------------------------------------------------
# AgentGraphState backwards compatibility
# ---------------------------------------------------------------------------


def test_state_schema_new_fields_have_empty_defaults():
    """New sensor fields should default to empty collections."""
    state = AgentGraphState()
    assert state.joint_positions == {}
    assert state.joint_velocities == {}
    assert state.imu_data == {}
    assert state.body_poses == {}
    assert state.contact_forces == []
    assert state.com_position == []
    assert state.zmp_position == []
    assert state.proprioceptive_state == {}


def test_state_schema_backwards_compatible():
    """Creating state with only old fields should work (no new fields required)."""
    from roboclaw.orchestration.state_schema import AgentGraphState

    state = AgentGraphState(
        robot_id="h1",
        task_spec={"goal": "test"},
    )
    assert state.robot_id == "h1"
    assert state.task_spec == {"goal": "test"}
    # New fields should be populated with defaults
    assert state.joint_positions == {}
    assert state.com_position == []
