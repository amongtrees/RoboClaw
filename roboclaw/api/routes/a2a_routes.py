"""A2A protocol HTTP endpoints.

Implements the Google Agent-to-Agent protocol:
- GET /.well-known/agent.json — AgentCard discovery
- POST /a2a/tasks — Submit a delegated task
- GET /a2a/tasks/{task_id} — Get delegated task status
- GET /a2a/tasks/{task_id}/stream — SSE stream of task progress
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

# Agent card for this agent
_agent_card = {
    "name": "RoboClaw Humanoid Agent",
    "description": "A humanoid robot agent capable of navigation, manipulation, and human interaction.",
    "url": "http://localhost:8000",
    "version": "0.2.0",
    "capabilities": [
        {"capability_id": "bipedal_locomotion", "description": "Bipedal walking and balance control"},
        {"capability_id": "dual_arm_manipulation", "description": "Dual-arm grasping and object manipulation"},
        {"capability_id": "navigation", "description": "Room-level navigation and path planning"},
        {"capability_id": "speech", "description": "Text-to-speech output"},
        {"capability_id": "scene_understanding", "description": "Visual scene perception and object detection"},
    ],
    "default_input_modes": ["text"],
    "default_output_modes": ["text"],
    "skills": [],
    "provider": "RoboClaw",
}

# In-memory A2A task store
_a2a_tasks: dict[str, dict] = {}


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Serve the AgentCard for A2A discovery."""
    return _agent_card


@router.post("/a2a/tasks")
async def submit_a2a_task(body: dict):
    """Accept a delegated task from a peer agent.

    Body:
        task_id: str
        description: str
        input_artifacts: list[str]
        deadline_sec: float | None
    """
    task_id = body.get("task_id", f"a2a_task_{uuid4().hex[:12]}")
    task = {
        "task_id": task_id,
        "description": body.get("description", ""),
        "input_artifacts": body.get("input_artifacts", []),
        "deadline_sec": body.get("deadline_sec"),
        "status": "accepted",
        "origin_agent": body.get("origin_agent", "unknown"),
        "created_at": None,
    }
    _a2a_tasks[task_id] = task
    return task


@router.get("/a2a/tasks/{task_id}")
async def get_a2a_task(task_id: str):
    """Get the status of a delegated task."""
    task = _a2a_tasks.get(task_id)
    if not task:
        return {"error": f"A2A task '{task_id}' not found"}, 404
    return task


@router.get("/a2a/tasks/{task_id}/stream")
async def stream_a2a_task(task_id: str):
    """SSE stream of delegated task progress."""

    async def event_generator():
        task = _a2a_tasks.get(task_id, {})
        yield {
            "event": "task_update",
            "data": json.dumps(task),
        }

    return EventSourceResponse(event_generator())
