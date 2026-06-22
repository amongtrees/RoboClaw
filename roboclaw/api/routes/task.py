"""Task management routes: submit, query, cancel tasks."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException

router = APIRouter()

# In-memory task store for Phase 1
_tasks: dict[str, dict] = {}


@router.post("/tasks")
async def submit_task(body: dict):
    """Submit a new task to a robot agent."""
    robot_id = body.get("robot_id", "")
    goal = body.get("goal", "")

    if not robot_id or not goal:
        raise HTTPException(status_code=400, detail="robot_id and goal are required")

    task_id = f"task_{uuid4().hex[:12]}"
    task = {
        "task_id": task_id,
        "robot_id": robot_id,
        "goal": goal,
        "priority": body.get("priority", 0),
        "deadline_sec": body.get("deadline_sec"),
        "status": "accepted",
        "phase": "idle",
        "created_at": None,
    }
    _tasks[task_id] = task
    return task


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and current phase."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@router.get("/tasks/{task_id}/plan")
async def get_task_plan(task_id: str):
    """Get the generated execution plan for a task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task.get("plan", {"status": "not_generated"})


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    if task["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Task already in terminal state: {task['status']}")

    task["status"] = "cancelled"
    return task


@router.get("/tasks")
async def list_tasks(robot_id: str | None = None, status: str | None = None):
    """List tasks, optionally filtered by robot_id and/or status."""
    results = list(_tasks.values())
    if robot_id:
        results = [t for t in results if t["robot_id"] == robot_id]
    if status:
        results = [t for t in results if t["status"] == status]
    return results
