"""Run a local RoboClaw memory demo."""

from pathlib import Path
from pprint import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboclaw import RobotMemoryAgent  # noqa: E402


def main() -> None:
    db_path = ROOT / ".roboclaw" / "demo.db"
    agent = RobotMemoryAgent(robot_id="ur5e_demo", db_path=db_path)

    agent.update_current_state(
        frame_id="base_link",
        camera_frame_id="camera_color_optical_frame",
        joint_states={"shoulder_pan_joint": 0.1, "wrist_3_joint": -0.2},
        end_effector_pose={
            "frame_id": "base_link",
            "position": [0.41, -0.12, 0.30],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        force_torque={
            "frame_id": "tool0",
            "force_xyz": [0.0, 0.0, 2.1],
            "torque_xyz": [0.0, 0.0, 0.0],
        },
        visible_objects=[
            {
                "label": "red_cube",
                "pose_frame": "camera_color_optical_frame",
                "confidence": 0.93,
            }
        ],
        safety_state={"estop": False, "workspace_ok": True},
    )

    episode_id = agent.start_episode(
        task_type="grasp",
        goal="pick red cube from the table",
        skill_name="top_down_grasp",
        metadata={"scene": "tabletop"},
    )

    agent.record_episode_event(
        episode_id,
        "navigation_context",
        {"base_frame": "base_link", "camera_frame": "camera_color_optical_frame"},
    )
    agent.record_episode_event(
        episode_id,
        "force_feedback",
        {"normal_force_n": 8.2, "slip_detected": False},
    )
    agent.add_artifact(
        artifact_type="trajectory_jsonl",
        uri=f"file://{ROOT}/.roboclaw/artifacts/{episode_id}/trajectory.jsonl",
        episode_id=episode_id,
        metadata={"format": "jsonl"},
    )

    agent.finalize_episode(
        episode_id,
        outcome="success",
        summary="Grasp succeeded with stable force feedback and no slip.",
        metadata={"force_feedback": "stable", "retry_count": 0},
    )

    print("Current state:")
    pprint(agent.get_current_state())

    print("\nSimilar grasp episodes:")
    pprint(agent.search_similar_episodes("stable force feedback grasp", task_type="grasp"))

    print("\nLast failure:")
    pprint(agent.explain_last_failure(task_type="grasp"))

    agent.close()


if __name__ == "__main__":
    main()
