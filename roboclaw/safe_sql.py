"""Safe read-only SQLite execution for model-generated SQL."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_TABLES = {
    "episodes": {
        "episode_id",
        "robot_id",
        "task_type",
        "goal",
        "skill_name",
        "started_at",
        "ended_at",
        "outcome",
        "summary",
        "initial_state_json",
        "final_state_json",
        "metadata_json",
        "searchable_text",
    },
    "episode_events": {
        "event_id",
        "episode_id",
        "event_type",
        "timestamp",
        "payload_json",
    },
    "artifacts": {
        "artifact_id",
        "artifact_type",
        "uri",
        "episode_id",
        "created_at",
        "metadata_json",
    },
}

DEFAULT_ALLOWED_FUNCTIONS = {
    "avg",
    "coalesce",
    "count",
    "date",
    "datetime",
    "ifnull",
    "julianday",
    "length",
    "lower",
    "max",
    "min",
    "round",
    "strftime",
    "substr",
    "sum",
    "time",
    "trim",
    "upper",
}


class UnsafeSQLQuery(ValueError):
    """Raised when a SQL query violates the read-only safety policy."""


class SafeSQLiteExecutor:
    """Execute untrusted Text-to-SQL output with a strict read-only policy."""

    def __init__(
        self,
        db_path: str | Path,
        allowed_tables: dict[str, set[str]] | None = None,
        allowed_functions: set[str] | None = None,
        max_rows: int = 100,
        timeout_sec: float = 2.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.allowed_tables = allowed_tables or DEFAULT_ALLOWED_TABLES
        self.allowed_functions = allowed_functions or DEFAULT_ALLOWED_FUNCTIONS
        self.max_rows = max_rows
        self.timeout_sec = timeout_sec

    def execute(self, sql: str) -> list[dict[str, Any]]:
        checked_sql = self._check_sql_shape(sql)
        started_at = time.monotonic()
        connection = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            timeout=self.timeout_sec,
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.enable_load_extension(False)
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(self._authorizer)

            def progress_handler() -> int:
                if time.monotonic() - started_at > self.timeout_sec:
                    return 1
                return 0

            connection.set_progress_handler(progress_handler, 1000)
            cursor = connection.execute(checked_sql)
            rows = cursor.fetchmany(self.max_rows)
            return [dict(row) for row in rows]
        except sqlite3.DatabaseError as exc:
            raise UnsafeSQLQuery(str(exc)) from exc
        finally:
            connection.close()

    def _check_sql_shape(self, sql: str) -> str:
        checked_sql = sql.strip().rstrip(";").strip()
        lowered = checked_sql.lower()
        if not checked_sql:
            raise UnsafeSQLQuery("SQL query is empty")
        if not (lowered.startswith("select ") or lowered.startswith("with ")):
            raise UnsafeSQLQuery("Only SELECT queries are allowed")
        if ";" in checked_sql:
            raise UnsafeSQLQuery("Only one SQL statement is allowed")
        return checked_sql

    def _authorizer(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger_or_view: str | None,
    ) -> int:
        del db_name, trigger_or_view
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            return self._authorize_read(arg1, arg2)
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (arg2 or arg1 or "").lower()
            if function_name in self.allowed_functions:
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _authorize_read(self, table_name: str | None, column_name: str | None) -> int:
        if table_name is None:
            return sqlite3.SQLITE_DENY
        allowed_columns = self.allowed_tables.get(table_name)
        if allowed_columns is None:
            return sqlite3.SQLITE_DENY
        if column_name in (None, "") or column_name in allowed_columns:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
