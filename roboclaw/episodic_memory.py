"""SQLite-backed episodic memory for robot task timelines."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from time import time
from typing import Any

from .models import ArtifactRef, EpisodeEvent, EpisodeRecord, Outcome


class EpisodicMemory:
    """Stores robot episodes, timeline events, and artifact references."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self._conn.close()

    def _initialize(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                robot_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                goal TEXT NOT NULL,
                skill_name TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                outcome TEXT NOT NULL,
                summary TEXT NOT NULL,
                initial_state_json TEXT NOT NULL,
                final_state_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                searchable_text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episode_events (
                event_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                uri TEXT NOT NULL,
                episode_id TEXT,
                created_at REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                    ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_robot_task
                ON episodes(robot_id, task_type, outcome, started_at);

            CREATE INDEX IF NOT EXISTS idx_events_episode_time
                ON episode_events(episode_id, timestamp);
            """
        )
        self._try_create_fts()
        self._conn.commit()

    def _try_create_fts(self) -> None:
        try:
            self._conn.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS episode_search
                USING fts5(episode_id UNINDEXED, searchable_text);
                """
            )
        except sqlite3.OperationalError:
            # Some Python SQLite builds do not include FTS5. LIKE fallback remains.
            pass

    def start_episode(
        self,
        robot_id: str,
        task_type: str,
        goal: str,
        skill_name: str | None = None,
        initial_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        episode = EpisodeRecord.create(
            robot_id=robot_id,
            task_type=task_type,
            goal=goal,
            skill_name=skill_name,
            initial_state=initial_state,
            metadata=metadata,
        )
        self._upsert_episode(episode)
        return episode

    def record_event(
        self,
        episode_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> EpisodeEvent:
        event = EpisodeEvent.create(
            episode_id=episode_id,
            event_type=event_type,
            payload=payload,
        )
        self._conn.execute(
            """
            INSERT INTO episode_events
                (event_id, episode_id, event_type, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.episode_id,
                event.event_type,
                event.timestamp,
                json.dumps(event.payload, sort_keys=True),
            ),
        )
        self._conn.commit()
        return event

    def finalize_episode(
        self,
        episode_id: str,
        outcome: Outcome,
        summary: str,
        final_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        episode = self.get_episode(episode_id)
        if episode is None:
            raise KeyError(f"Episode not found: {episode_id}")

        episode.outcome = outcome
        episode.summary = summary
        episode.ended_at = time()
        episode.final_state = final_state or {}
        if metadata:
            episode.metadata.update(metadata)
        self._upsert_episode(episode)
        self.record_event(
            episode_id,
            "episode_finalized",
            {"outcome": outcome, "summary": summary, "metadata": metadata or {}},
        )
        return episode

    def add_artifact(
        self,
        artifact_type: str,
        uri: str,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        artifact = ArtifactRef.create(
            artifact_type=artifact_type,
            uri=uri,
            episode_id=episode_id,
            metadata=metadata,
        )
        self._conn.execute(
            """
            INSERT INTO artifacts
                (artifact_id, artifact_type, uri, episode_id, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                artifact.artifact_type,
                artifact.uri,
                artifact.episode_id,
                artifact.created_at,
                json.dumps(artifact.metadata, sort_keys=True),
            ),
        )
        self._conn.commit()
        return artifact

    def get_episode(self, episode_id: str) -> EpisodeRecord | None:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        return self._row_to_episode(row) if row else None

    def list_events(self, episode_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM episode_events
            WHERE episode_id = ?
            ORDER BY timestamp ASC
            """,
            (episode_id,),
        ).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def search_episodes(
        self,
        query: str,
        task_type: str | None = None,
        outcome: Outcome | None = None,
        robot_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        filters: list[str] = []
        params: list[Any] = []
        if task_type:
            filters.append("e.task_type = ?")
            params.append(task_type)
        if outcome:
            filters.append("e.outcome = ?")
            params.append(outcome)
        if robot_id:
            filters.append("e.robot_id = ?")
            params.append(robot_id)

        where = " AND ".join(filters)
        where_clause = f"WHERE {where}" if where else ""

        if query and self._has_fts():
            sql = f"""
                SELECT e.*
                FROM episode_search s
                JOIN episodes e ON e.episode_id = s.episode_id
                {where_clause + " AND" if where_clause else "WHERE"}
                    episode_search MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = self._conn.execute(sql, (*params, query, limit)).fetchall()
        else:
            like_query = f"%{query}%"
            sql = f"""
                SELECT e.*
                FROM episodes e
                {where_clause + " AND" if where_clause else "WHERE"}
                    e.searchable_text LIKE ?
                ORDER BY e.started_at DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, (*params, like_query, limit)).fetchall()

        return [self._episode_with_counts(row) for row in rows]

    def explain_last_failure(
        self,
        task_type: str | None = None,
        robot_id: str | None = None,
    ) -> dict[str, Any] | None:
        filters = ["outcome IN ('failure', 'blocked', 'aborted')"]
        params: list[Any] = []
        if task_type:
            filters.append("task_type = ?")
            params.append(task_type)
        if robot_id:
            filters.append("robot_id = ?")
            params.append(robot_id)
        sql = f"""
            SELECT *
            FROM episodes
            WHERE {" AND ".join(filters)}
            ORDER BY ended_at DESC, started_at DESC
            LIMIT 1
        """
        row = self._conn.execute(sql, params).fetchone()
        if not row:
            return None
        episode = self._episode_with_counts(row)
        episode["events"] = self.list_events(row["episode_id"])
        return episode

    def _upsert_episode(self, episode: EpisodeRecord) -> None:
        searchable_text = episode.searchable_text()
        self._conn.execute(
            """
            INSERT INTO episodes (
                episode_id, robot_id, task_type, goal, skill_name, started_at,
                ended_at, outcome, summary, initial_state_json, final_state_json,
                metadata_json, searchable_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                ended_at = excluded.ended_at,
                outcome = excluded.outcome,
                summary = excluded.summary,
                final_state_json = excluded.final_state_json,
                metadata_json = excluded.metadata_json,
                searchable_text = excluded.searchable_text
            """,
            (
                episode.episode_id,
                episode.robot_id,
                episode.task_type,
                episode.goal,
                episode.skill_name,
                episode.started_at,
                episode.ended_at,
                episode.outcome,
                episode.summary,
                json.dumps(episode.initial_state, sort_keys=True),
                json.dumps(episode.final_state, sort_keys=True),
                json.dumps(episode.metadata, sort_keys=True),
                searchable_text,
            ),
        )
        if self._has_fts():
            self._conn.execute(
                "DELETE FROM episode_search WHERE episode_id = ?",
                (episode.episode_id,),
            )
            self._conn.execute(
                """
                INSERT INTO episode_search (episode_id, searchable_text)
                VALUES (?, ?)
                """,
                (episode.episode_id, searchable_text),
            )
        self._conn.commit()

    def _has_fts(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episode_search'"
        ).fetchone()
        return row is not None

    def _row_to_episode(self, row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=row["episode_id"],
            robot_id=row["robot_id"],
            task_type=row["task_type"],
            goal=row["goal"],
            skill_name=row["skill_name"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            outcome=row["outcome"],
            summary=row["summary"],
            initial_state=json.loads(row["initial_state_json"]),
            final_state=json.loads(row["final_state_json"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def _episode_with_counts(self, row: sqlite3.Row) -> dict[str, Any]:
        episode = self._row_to_episode(row).to_dict()
        event_count = self._conn.execute(
            "SELECT COUNT(*) FROM episode_events WHERE episode_id = ?",
            (row["episode_id"],),
        ).fetchone()[0]
        artifact_count = self._conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE episode_id = ?",
            (row["episode_id"],),
        ).fetchone()[0]
        episode["event_count"] = event_count
        episode["artifact_count"] = artifact_count
        return episode

    @staticmethod
    def _event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "episode_id": row["episode_id"],
            "event_type": row["event_type"],
            "timestamp": row["timestamp"],
            "payload": json.loads(row["payload_json"]),
        }
