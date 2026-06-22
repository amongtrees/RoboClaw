"""Agent lifecycle routes: register, status, mode management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

# In-memory agent store for Phase 1 (production uses PostgreSQL)
_agents: dict[str, dict] = {}


@router.post("/agents")
async def register_agent(body: dict):
    """Register a new robot agent."""
    robot_id = body.get("robot_id", "")
    if not robot_id:
        raise HTTPException(status_code=400, detail="robot_id is required")

    agent_info = {
        "robot_id": robot_id,
        "robot_model": body.get("robot_model", "unknown"),
        "display_name": body.get("display_name", robot_id),
        "status": "idle",
        "mode": body.get("mode", "autonomous"),
        "registered_at": None,
    }
    _agents[robot_id] = agent_info
    return agent_info


@router.get("/agents/{robot_id}")
async def get_agent(robot_id: str):
    """Get agent info and current status."""
    agent = _agents.get(robot_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{robot_id}' not found")
    return agent


@router.put("/agents/{robot_id}/mode")
async def set_agent_mode(robot_id: str, body: dict):
    """Set agent operating mode: idle | autonomous | teleop | diagnostic | emergency_stop."""
    agent = _agents.get(robot_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{robot_id}' not found")

    mode = body.get("mode", "autonomous")
    valid_modes = {"idle", "autonomous", "teleop", "diagnostic", "emergency_stop"}
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")

    agent["mode"] = mode
    return agent


@router.delete("/agents/{robot_id}")
async def decommission_agent(robot_id: str):
    """Decommission an agent."""
    if robot_id not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent '{robot_id}' not found")
    del _agents[robot_id]
    return {"status": "decommissioned", "robot_id": robot_id}


@router.get("/agents")
async def list_agents():
    """List all registered agents."""
    return list(_agents.values())
