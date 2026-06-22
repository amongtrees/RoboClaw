"""Telemetry routes: state snapshot, SSE streaming, WebSocket endpoints."""

from __future__ import annotations

import asyncio
import json
from time import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

# In-memory state store for Phase 1
_robot_states: dict[str, dict] = {}


# --- State Snapshot ---

@router.get("/telemetry/{robot_id}/state")
async def get_robot_state(robot_id: str):
    """Get the latest robot state snapshot."""
    state = _robot_states.get(robot_id)
    if not state:
        return {"error": f"No state available for robot '{robot_id}'"}, 404
    return state


# --- SSE Streaming ---

@router.get("/telemetry/{robot_id}/stream")
async def stream_robot_state(robot_id: str):
    """SSE stream of robot state updates."""

    async def event_generator():
        while True:
            state = _robot_states.get(robot_id, {})
            yield {
                "event": "state_update",
                "data": json.dumps({
                    "robot_id": robot_id,
                    "timestamp": time(),
                    **state,
                }),
            }
            await asyncio.sleep(0.1)  # 10 Hz default

    return EventSourceResponse(event_generator())


# --- WebSocket ---

@router.websocket("/ws/{robot_id}/telemetry")
async def websocket_telemetry(websocket: WebSocket, robot_id: str):
    """WebSocket: push robot state every 100ms."""
    await websocket.accept()
    try:
        while True:
            state = _robot_states.get(robot_id, {})
            await websocket.send_json({
                "type": "state_update",
                "robot_id": robot_id,
                "timestamp": time(),
                "data": state,
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@router.websocket("/ws/{robot_id}/command")
async def websocket_command(websocket: WebSocket, robot_id: str):
    """WebSocket: bidirectional command/response channel."""
    await websocket.accept()
    try:
        # Send initial ack
        await websocket.send_json({
            "type": "connected",
            "robot_id": robot_id,
            "message": "Command channel established",
        })

        while True:
            data = await websocket.receive_json()
            command = data.get("command", "")
            cmd_type = data.get("type", "")

            if command == "pause":
                await websocket.send_json({"type": "ack", "command": "pause", "status": "accepted"})
            elif command == "resume":
                await websocket.send_json({"type": "ack", "command": "resume", "status": "accepted"})
            elif command == "estop":
                await websocket.send_json({"type": "ack", "command": "estop", "status": "accepted", "warning": "Emergency stop activated"})
            elif cmd_type == "hri_message":
                text = data.get("text", "")
                await websocket.send_json({"type": "hri_response", "text": f"Received: {text}"})
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown command: {command}"})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
