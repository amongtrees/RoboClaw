"""Tool-like facade for RoboClaw robot memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .episodic_memory import EpisodicMemory
from .models import Outcome
from .working_memory import WorkingMemory


class RobotMemoryAgent:
    """Coordinates working memory and episodic memory for one robot."""

    def __init__(
        self,
        robot_id: str,
        db_path: str | Path = ".roboclaw/roboclaw.db",
        working_ttl_sec: float = 5.0,
    ) -> None:
        self.robot_id = robot_id
        self.working_memory = WorkingMemory(
            robot_id=robot_id,
            ttl_sec=working_ttl_sec,
        )
        self.episodic_memory = EpisodicMemory(db_path=db_path)

    def close(self) -> None:
        self.episodic_memory.close()

    def update_current_state(self, **state_updates: Any) -> dict[str, Any]:
        state = self.working_memory.update_state(**state_updates)
        return state.to_dict()

    def get_current_state(self) -> dict[str, Any]:
        return self.working_memory.get_state()

    def start_episode(
        self,
        task_type: str,
        goal: str,
        skill_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        initial_state = self.get_current_state()
        episode = self.episodic_memory.start_episode(
            robot_id=self.robot_id,
            task_type=task_type,
            goal=goal,
            skill_name=skill_name,
            initial_state=initial_state,
            metadata=metadata,
        )
        self.working_memory.update_state(
            task_id=episode.episode_id,
            skill_name=skill_name,
            phase="started",
        )
        self.episodic_memory.record_event(
            episode.episode_id,
            "episode_started",
            {"goal": goal, "skill_name": skill_name},
        )
        return episode.episode_id

    def record_episode_event(
        self,
        episode_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event = self.episodic_memory.record_event(
            episode_id=episode_id,
            event_type=event_type,
            payload=payload,
        )
        self.working_memory.add_event(event_type, payload)
        return event.event_id

    def finalize_episode(
        self,
        episode_id: str,
        outcome: Outcome,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        final_state = self.get_current_state()
        episode = self.episodic_memory.finalize_episode(
            episode_id=episode_id,
            outcome=outcome,
            summary=summary,
            final_state=final_state,
            metadata=metadata,
        )
        self.working_memory.update_state(task_id=None, phase="idle")
        return episode.to_dict()

    def add_artifact(
        self,
        artifact_type: str,
        uri: str,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        artifact = self.episodic_memory.add_artifact(
            artifact_type=artifact_type,
            uri=uri,
            episode_id=episode_id,
            metadata=metadata,
        )
        return artifact.artifact_id

    def search_similar_episodes(
        self,
        query: str,
        task_type: str | None = None,
        outcome: Outcome | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.episodic_memory.search_episodes(
            query=query,
            task_type=task_type,
            outcome=outcome,
            robot_id=self.robot_id,
            limit=limit,
        )

    def explain_last_failure(
        self,
        task_type: str | None = None,
    ) -> dict[str, Any] | None:
        return self.episodic_memory.explain_last_failure(
            task_type=task_type,
            robot_id=self.robot_id,
        )
