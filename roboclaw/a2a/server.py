"""A2A server — serves agent card, accepts delegated tasks, streams progress."""

from __future__ import annotations

from typing import Any


class A2AServer:
    """Server-side A2A protocol handler.

    Exposes:
    - AgentCard at /.well-known/agent.json
    - Task submission endpoint
    - Task status polling
    - SSE progress streaming
    """

    def __init__(self, agent_card: dict[str, Any] | None = None) -> None:
        self._agent_card = agent_card or {}
        self._tasks: dict[str, dict[str, Any]] = {}

    @property
    def agent_card(self) -> dict[str, Any]:
        return self._agent_card

    async def accept_task(self, task_descriptor: dict[str, Any]) -> str:
        """Accept a delegated task from a peer agent.

        Returns the task_id.
        """
        task_id = task_descriptor.get("task_id", "")
        self._tasks[task_id] = {
            **task_descriptor,
            "status": "accepted",
        }
        return task_id

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get the current status of a delegated task."""
        return self._tasks.get(task_id)

    async def update_task_status(self, task_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        """Update a task's status and optionally set the result."""
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = status
            if result:
                self._tasks[task_id]["result"] = result

    async def get_pending_tasks(self) -> list[dict[str, Any]]:
        """Get all pending/accepted tasks."""
        return [
            t for t in self._tasks.values()
            if t.get("status") in ("pending", "accepted")
        ]
