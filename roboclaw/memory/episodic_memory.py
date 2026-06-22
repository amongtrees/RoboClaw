"""Episodic memory — long-term storage of task execution episodes in PostgreSQL.

Evolves the SQLite-based EpisodicMemory to PostgreSQL with asyncpg/SQLAlchemy 2.0.
Provides full CRUD, text search, and filtering for past episodes.
"""

from __future__ import annotations

import json
from time import time
from typing import Any

from roboclaw.core.types import Outcome
from roboclaw.models.episode import Artifact, EpisodeEvent, EpisodeRecord


class EpisodicMemory:
    """PostgreSQL-backed episodic memory store.

    In production, this uses SQLAlchemy 2.0 async sessions with asyncpg.
    For development/testing without PostgreSQL, it falls back to an in-memory dict.
    """

    def __init__(self, db_session_factory: Any = None) -> None:
        self._session_factory = db_session_factory
        # In-memory fallback for development/testing
        self._episodes: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Set up database tables if using PostgreSQL."""
        if self._session_factory:
            # Tables are created via Alembic migrations in production
            pass

    # --- Episode CRUD ---

    async def save_episode(self, episode: EpisodeRecord) -> str:
        """Save a new episode record. Returns the episode_id."""
        data = episode.to_dict()
        self._episodes[episode.episode_id] = data

        if self._session_factory:
            async with self._session_factory() as session:
                await session.execute(
                    """
                    INSERT INTO episodes (episode_id, robot_id, task_id, task_type, goal,
                        skill_name, plan_id, started_at, ended_at, outcome, summary,
                        initial_state_json, final_state_json, metadata_json,
                        recovery_attempts, total_duration_sec)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        ended_at = $9, outcome = $10, summary = $11,
                        final_state_json = $13, total_duration_sec = $16
                    """,
                    episode.episode_id,
                    episode.robot_id,
                    episode.task_id,
                    episode.task_type,
                    episode.goal,
                    episode.skill_name,
                    episode.plan_id,
                    episode.started_at,
                    episode.ended_at,
                    episode.outcome,
                    episode.summary,
                    json.dumps(episode.initial_state),
                    json.dumps(episode.final_state),
                    json.dumps(episode.metadata),
                    episode.recovery_attempts,
                    episode.total_duration_sec,
                )
                await session.commit()

        return episode.episode_id

    async def get_episode(self, episode_id: str) -> EpisodeRecord | None:
        """Get an episode by ID."""
        if self._session_factory:
            async with self._session_factory() as session:
                result = await session.execute(
                    "SELECT * FROM episodes WHERE episode_id = $1", episode_id
                )
                row = result.fetchone()
                if row:
                    return self._row_to_episode(row)

        data = self._episodes.get(episode_id)
        if data:
            return EpisodeRecord(**data)
        return None

    async def search_episodes(
        self,
        robot_id: str | None = None,
        task_type: str | None = None,
        outcome: Outcome | None = None,
        query_text: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EpisodeRecord]:
        """Search episodes with filters and optional full-text search."""
        if self._session_factory:
            conditions = []
            params: list[Any] = []
            param_idx = 1

            if robot_id:
                conditions.append(f"robot_id = ${param_idx}")
                params.append(robot_id)
                param_idx += 1
            if task_type:
                conditions.append(f"task_type = ${param_idx}")
                params.append(task_type)
                param_idx += 1
            if outcome:
                conditions.append(f"outcome = ${param_idx}")
                params.append(outcome)
                param_idx += 1
            if query_text:
                conditions.append(
                    f"to_tsvector('english', goal || ' ' || summary) @@ plainto_tsquery('english', ${param_idx})"
                )
                params.append(query_text)
                param_idx += 1

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params.extend([limit, offset])

            async with self._session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM episodes WHERE {where_clause} ORDER BY started_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}",
                    *params,
                )
                return [self._row_to_episode(row) for row in result.fetchall()]

        # In-memory fallback
        results = []
        for data in self._episodes.values():
            if robot_id and data.get("robot_id") != robot_id:
                continue
            if task_type and data.get("task_type") != task_type:
                continue
            if outcome and data.get("outcome") != outcome:
                continue
            if query_text:
                text = f"{data.get('goal', '')} {data.get('summary', '')}"
                if query_text.lower() not in text.lower():
                    continue
            results.append(EpisodeRecord(**data))

        results.sort(key=lambda e: e.started_at, reverse=True)
        return results[offset : offset + limit]

    # --- Events ---

    async def save_event(self, event: EpisodeEvent) -> str:
        """Save an episode event."""
        data = event.to_dict()
        self._events.setdefault(event.episode_id, []).append(data)

        if self._session_factory:
            async with self._session_factory() as session:
                await session.execute(
                    """
                    INSERT INTO episode_events (event_id, episode_id, event_type, timestamp, payload_json)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    event.event_id,
                    event.episode_id,
                    event.event_type,
                    event.timestamp,
                    json.dumps(event.payload),
                )
                await session.commit()

        return event.event_id

    async def get_events(self, episode_id: str) -> list[EpisodeEvent]:
        """Get all events for an episode, ordered by timestamp."""
        if self._session_factory:
            async with self._session_factory() as session:
                result = await session.execute(
                    "SELECT * FROM episode_events WHERE episode_id = $1 ORDER BY timestamp",
                    episode_id,
                )
                return [
                    EpisodeEvent(
                        event_id=row["event_id"],
                        episode_id=row["episode_id"],
                        event_type=row["event_type"],
                        timestamp=row["timestamp"],
                        payload=row["payload_json"],
                    )
                    for row in result.fetchall()
                ]

        events_data = self._events.get(episode_id, [])
        return [EpisodeEvent(**d) for d in sorted(events_data, key=lambda d: d.get("timestamp", 0))]

    # --- Artifacts ---

    async def save_artifact(self, artifact: Artifact) -> str:
        """Save an artifact reference."""
        data = artifact.to_dict()
        self._artifacts[artifact.artifact_id] = data

        if self._session_factory:
            async with self._session_factory() as session:
                await session.execute(
                    """
                    INSERT INTO artifacts (artifact_id, artifact_type, uri, episode_id, created_at, size_bytes, metadata_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    artifact.artifact_id,
                    artifact.artifact_type,
                    artifact.uri,
                    artifact.episode_id,
                    artifact.created_at,
                    artifact.size_bytes,
                    json.dumps(artifact.metadata),
                )
                await session.commit()

        return artifact.artifact_id

    # --- Helpers ---

    def _row_to_episode(self, row: Any) -> EpisodeRecord:
        """Convert a database row to an EpisodeRecord."""
        return EpisodeRecord(
            episode_id=row["episode_id"],
            robot_id=row["robot_id"],
            task_id=row.get("task_id"),
            task_type=row["task_type"],
            goal=row["goal"],
            skill_name=row.get("skill_name"),
            plan_id=row.get("plan_id"),
            started_at=row["started_at"],
            ended_at=row.get("ended_at"),
            outcome=row["outcome"],
            summary=row.get("summary", ""),
            initial_state=row.get("initial_state_json", {}),
            final_state=row.get("final_state_json", {}),
            metadata=row.get("metadata_json", {}),
            recovery_attempts=row.get("recovery_attempts", 0),
            total_duration_sec=row.get("total_duration_sec"),
        )

    async def close(self) -> None:
        """Clean up resources."""
        pass
