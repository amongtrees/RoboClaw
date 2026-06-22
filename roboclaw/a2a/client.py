"""A2A client — discovers peers, submits tasks, polls/streams results."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for communicating with peer agents via the A2A protocol.

    Discovers peer agents, submits tasks for delegation, and retrieves results.
    """

    def __init__(self, httpx_client: Any = None) -> None:
        self._http = httpx_client
        self._pending_tasks: dict[str, dict[str, Any]] = {}

    async def discover_agent(self, agent_url: str) -> dict[str, Any] | None:
        """Fetch an agent's AgentCard from its well-known endpoint.

        Args:
            agent_url: Base URL of the peer agent (e.g., http://agent:8000).

        Returns:
            AgentCard dict, or None if unreachable.
        """
        try:
            if self._http:
                response = await self._http.get(f"{agent_url}/.well-known/agent.json")
                response.raise_for_status()
                return response.json()
            else:
                # Stub: simulate discovery
                logger.info(f"Discovered agent at {agent_url} (stub)")
                return {
                    "name": f"Agent@{agent_url}",
                    "url": agent_url,
                    "capabilities": [],
                }
        except Exception as e:
            logger.warning(f"Failed to discover agent at {agent_url}: {e}")
            return None

    async def submit_task(
        self,
        agent_url: str,
        task_description: str,
        input_artifacts: list[str] | None = None,
    ) -> str | None:
        """Submit a task to a peer agent for execution.

        Args:
            agent_url: The peer agent's base URL.
            task_description: Natural language description of the task.
            input_artifacts: Optional list of artifact IDs to pass along.

        Returns:
            task_id if accepted, None if submission failed.
        """
        task_payload = {
            "description": task_description,
            "input_artifacts": input_artifacts or [],
            "origin_agent": "roboclaw",
        }

        try:
            if self._http:
                response = await self._http.post(
                    f"{agent_url}/a2a/tasks",
                    json=task_payload,
                )
                response.raise_for_status()
                result = response.json()
                task_id = result.get("task_id", "")
                self._pending_tasks[task_id] = {
                    "agent_url": agent_url,
                    "status": "submitted",
                }
                return task_id
            else:
                # Stub: simulate submission
                import uuid
                task_id = f"a2a_{uuid.uuid4().hex[:12]}"
                logger.info(f"Submitted task to {agent_url}: {task_description} (stub, task_id={task_id})")
                self._pending_tasks[task_id] = {
                    "agent_url": agent_url,
                    "status": "simulated_accepted",
                }
                return task_id
        except Exception as e:
            logger.error(f"Failed to submit task to {agent_url}: {e}")
            return None

    async def get_task_result(self, task_id: str) -> dict[str, Any] | None:
        """Poll for the result of a delegated task.

        Returns None if the task is still in progress.
        """
        task_info = self._pending_tasks.get(task_id)
        if not task_info:
            return None

        agent_url = task_info["agent_url"]

        try:
            if self._http:
                response = await self._http.get(f"{agent_url}/a2a/tasks/{task_id}")
                response.raise_for_status()
                return response.json()
            else:
                # Stub: simulate completed result
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": {"simulated": True},
                }
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return None
