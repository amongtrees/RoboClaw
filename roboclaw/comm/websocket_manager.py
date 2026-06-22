"""WebSocket connection manager — tracks and broadcasts to connected clients."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for telemetry and command channels.

    Supports:
    - Per-robot telemetry channels (one-to-many broadcast)
    - Per-session command channels (one-to-one bidirectional)
    """

    def __init__(self) -> None:
        # robot_id -> set of WebSocket connections
        self._telemetry_channels: dict[str, set[WebSocket]] = {}
        # session_id -> WebSocket connection
        self._command_channels: dict[str, WebSocket] = {}

    async def connect_telemetry(self, robot_id: str, websocket: WebSocket) -> None:
        """Register a telemetry WebSocket connection for a robot."""
        await websocket.accept()
        self._telemetry_channels.setdefault(robot_id, set()).add(websocket)
        logger.info(f"Telemetry WebSocket connected for robot '{robot_id}' (total: {len(self._telemetry_channels[robot_id])})")

    async def disconnect_telemetry(self, robot_id: str, websocket: WebSocket) -> None:
        """Remove a telemetry WebSocket connection."""
        if robot_id in self._telemetry_channels:
            self._telemetry_channels[robot_id].discard(websocket)
            if not self._telemetry_channels[robot_id]:
                del self._telemetry_channels[robot_id]

    async def broadcast_telemetry(self, robot_id: str, data: dict[str, Any]) -> None:
        """Broadcast telemetry data to all clients watching a robot."""
        channels = self._telemetry_channels.get(robot_id, set())
        disconnected = set()
        for ws in channels:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            await self.disconnect_telemetry(robot_id, ws)

    async def connect_command(self, session_id: str, websocket: WebSocket) -> None:
        """Register a command WebSocket connection."""
        await websocket.accept()
        self._command_channels[session_id] = websocket
        logger.info(f"Command WebSocket connected for session '{session_id}'")

    async def disconnect_command(self, session_id: str) -> None:
        """Remove a command WebSocket connection."""
        self._command_channels.pop(session_id, None)

    async def send_command_response(self, session_id: str, data: dict[str, Any]) -> None:
        """Send a response on a command channel."""
        ws = self._command_channels.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                await self.disconnect_command(session_id)

    @property
    def active_telemetry_count(self) -> int:
        return sum(len(v) for v in self._telemetry_channels.values())

    @property
    def active_command_count(self) -> int:
        return len(self._command_channels)
