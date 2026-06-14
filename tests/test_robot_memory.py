import tempfile
import unittest
from pathlib import Path

from roboclaw import RobotMemoryAgent


class RobotMemoryAgentTest(unittest.TestCase):
    def test_episode_lifecycle_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            agent = RobotMemoryAgent(robot_id="test_bot", db_path=db_path)

            agent.update_current_state(
                frame_id="base_link",
                camera_frame_id="camera",
                end_effector_pose={
                    "frame_id": "base_link",
                    "position": [1, 2, 3],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            )

            episode_id = agent.start_episode(
                task_type="grasp",
                goal="grasp red cube",
                skill_name="test_grasp",
            )
            event_id = agent.record_episode_event(
                episode_id,
                "force_feedback",
                {"normal_force_n": 5.0},
            )
            self.assertTrue(event_id.startswith("event_"))

            final_episode = agent.finalize_episode(
                episode_id,
                outcome="success",
                summary="Stable grasp with normal force feedback.",
            )
            self.assertEqual(final_episode["outcome"], "success")

            results = agent.search_similar_episodes("force feedback", task_type="grasp")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["episode_id"], episode_id)
            self.assertGreaterEqual(results[0]["event_count"], 2)

            agent.close()

    def test_last_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            agent = RobotMemoryAgent(robot_id="test_bot", db_path=db_path)
            episode_id = agent.start_episode(
                task_type="navigation",
                goal="move to shelf",
                skill_name="nav_to_pose",
            )
            agent.record_episode_event(
                episode_id,
                "safety_event",
                {"reason": "workspace boundary"},
            )
            agent.finalize_episode(
                episode_id,
                outcome="blocked",
                summary="Navigation blocked by workspace boundary.",
                metadata={"root_cause": "workspace boundary"},
            )

            failure = agent.explain_last_failure(task_type="navigation")
            self.assertIsNotNone(failure)
            self.assertEqual(failure["episode_id"], episode_id)
            self.assertEqual(failure["outcome"], "blocked")
            self.assertEqual(failure["events"][-1]["event_type"], "episode_finalized")

            agent.close()


if __name__ == "__main__":
    unittest.main()
