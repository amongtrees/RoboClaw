"""Unit tests for Pydantic models."""

import pytest

from roboclaw.models.agent import AgentCapability, AgentCard
from roboclaw.models.episode import Artifact, EpisodeEvent, EpisodeRecord
from roboclaw.models.perception import DetectedObject, PerceptionSnapshot
from roboclaw.models.plan import ExecutionPlan, SubTask
from roboclaw.models.pose import BipedalGaitParams, Pose, Twist
from roboclaw.models.safety import SafetyViolation, SafetyZone
from roboclaw.models.sensor import ForceTorque, IMUReading, ProprioceptiveState
from roboclaw.models.state import HumanoidState, RobotState
from roboclaw.models.task import TaskGraph, TaskSpec


class TestPose:
    def test_pose_creation(self):
        pose = Pose(frame_id="base_link", position=(1.0, 2.0, 3.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0))
        assert pose.frame_id == "base_link"
        assert pose.position == (1.0, 2.0, 3.0)

    def test_pose_from_dict(self):
        pose = Pose.from_dict({
            "frame_id": "camera",
            "position": [0.5, 0.3, 1.2],
            "orientation_xyzw": [0, 0, 0, 1],
        })
        assert pose.frame_id == "camera"
        assert pose.position == (0.5, 0.3, 1.2)

    def test_gait_params_defaults(self):
        gait = BipedalGaitParams()
        assert gait.step_length_m == 0.3
        assert gait.step_height_m == 0.05


class TestForceTorque:
    def test_force_torque_creation(self):
        ft = ForceTorque(force_xyz=(1.0, 0.0, 0.0), torque_xyz=(0.0, 0.0, 0.1))
        assert ft.force_xyz == (1.0, 0.0, 0.0)

    def test_force_torque_from_dict(self):
        ft = ForceTorque.from_dict({
            "sensor_id": "wrist_ft",
            "frame_id": "ee_link",
            "force_xyz": [5.0, 0.0, 0.0],
            "torque_xyz": [0.0, 0.0, 0.5],
        })
        assert ft.force_xyz == (5.0, 0.0, 0.0)
        assert ft.sensor_id == "wrist_ft"


class TestRobotState:
    def test_robot_state_creation(self):
        state = RobotState(robot_id="test_robot", phase="idle")
        assert state.robot_id == "test_robot"
        assert state.phase == "idle"
        assert state.joint_states == {}

    def test_humanoid_state(self):
        state = HumanoidState(
            robot_id="h1_test",
            zmp=(0.01, 0.02),
            support_phase="double",
            head_pan_rad=0.5,
        )
        assert state.zmp == (0.01, 0.02)
        assert state.support_phase == "double"
        assert state.head_pan_rad == 0.5


class TestEpisode:
    def test_episode_record_create(self):
        ep = EpisodeRecord.create(
            robot_id="test_robot",
            task_type="grasp",
            goal="pick up the cup",
        )
        assert ep.robot_id == "test_robot"
        assert ep.task_type == "grasp"
        assert ep.outcome == "unknown"
        assert ep.episode_id.startswith("episode_")

    def test_episode_finalize(self):
        ep = EpisodeRecord.create(robot_id="test", task_type="grasp", goal="test")
        ep.finalize("success", "Grasped successfully")
        assert ep.outcome == "success"
        assert ep.summary == "Grasped successfully"
        assert ep.ended_at is not None
        assert ep.total_duration_sec is not None

    def test_episode_searchable_text(self):
        ep = EpisodeRecord.create(
            robot_id="h1",
            task_type="navigation",
            goal="go to kitchen",
            metadata={"room": "kitchen"},
        )
        text = ep.searchable_text()
        assert "h1" in text
        assert "navigation" in text
        assert "go to kitchen" in text

    def test_artifact_create(self):
        art = Artifact.create("trajectory_jsonl", "file:///data/traj.jsonl")
        assert art.artifact_type == "trajectory_jsonl"
        assert art.artifact_id.startswith("artifact_")


class TestTask:
    def test_task_spec(self):
        spec = TaskSpec(robot_id="h1", goal="test goal", priority=1)
        assert spec.robot_id == "h1"
        assert spec.goal == "test goal"
        assert spec.priority == 1

    def test_sub_task_dependencies_satisfied(self):
        st = SubTask(
            sub_task_id="task_2",
            skill_type="walk_steps",
            preconditions=["task_1"],
        )
        assert st.dependencies_satisfied({"task_1"})
        assert not st.dependencies_satisfied(set())

    def test_task_graph_next_executable(self):
        graph = TaskGraph(sub_tasks=[
            SubTask(sub_task_id="a", skill_type="gaze_at", preconditions=[]),
            SubTask(sub_task_id="b", skill_type="walk_steps", preconditions=["a"]),
            SubTask(sub_task_id="c", skill_type="whole_body_grasp", preconditions=["b"]),
        ])

        next_tasks = graph.next_executable(set())
        assert len(next_tasks) == 1
        assert next_tasks[0].sub_task_id == "a"

        next_tasks = graph.next_executable({"a"})
        assert len(next_tasks) == 1
        assert next_tasks[0].sub_task_id == "b"

        assert graph.is_complete({"a", "b", "c"})
        assert not graph.is_complete({"a"})

    def test_execution_plan_progress(self):
        plan = ExecutionPlan(
            plan_id="test_plan",
            sub_tasks=[
                SubTask(sub_task_id="a", skill_type="gaze_at", preconditions=[]),
                SubTask(sub_task_id="b", skill_type="walk_steps", preconditions=["a"]),
            ],
        )
        assert plan.progress(set()) == 0.0
        assert plan.progress({"a"}) == 0.5
        assert plan.progress({"a", "b"}) == 1.0


class TestPerception:
    def test_detected_object(self):
        obj = DetectedObject(
            label="cup",
            confidence=0.95,
            position_world=(1.0, 0.5, 1.2),
        )
        assert obj.label == "cup"
        assert obj.confidence == 0.95

    def test_perception_snapshot(self):
        snap = PerceptionSnapshot(
            timestamp=123456.0,
            detected_objects=[
                DetectedObject(label="cup", confidence=0.9),
            ],
            speech_text="hello",
        )
        assert len(snap.detected_objects) == 1
        assert snap.speech_text == "hello"


class TestSafety:
    def test_safety_violation(self):
        sv = SafetyViolation(
            robot_id="h1",
            violation_type="force_limit",
            severity="warning",
            context={"force_n": 55.0, "limit_n": 50.0},
        )
        assert sv.robot_id == "h1"
        assert sv.violation_type == "force_limit"
        assert sv.severity == "warning"

    def test_safety_zone(self):
        zone = SafetyZone(
            zone_id="keep_out_1",
            zone_type="keep_out",
            dimensions={"x": 1.0, "y": 1.0, "z": 2.0},
        )
        assert zone.zone_type == "keep_out"


class TestAgentCard:
    def test_agent_card(self):
        card = AgentCard(
            name="Test Agent",
            description="A test humanoid robot",
            url="http://localhost:8000",
            capabilities=[
                AgentCapability(capability_id="navigation", description="Can navigate"),
            ],
        )
        assert card.name == "Test Agent"
        assert len(card.capabilities) == 1
        assert card.default_input_modes == ["text"]
