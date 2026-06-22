"""Spatial memory — metric-semantic map for environment understanding.

Maintains a 3D occupancy grid + semantic layer with room labels, object locations,
and navigable areas. In Phase 1, this is an in-memory structure.
"""

from __future__ import annotations

from typing import Any


class SpatialMemory:
    """Metric-semantic spatial map of the environment.

    Stores room topology, object locations, navigable areas, and occupancy data.
    """

    def __init__(self, resolution_m: float = 0.05) -> None:
        self.resolution_m = resolution_m
        # Room graph
        self._rooms: dict[str, dict[str, Any]] = {}
        self._room_connections: list[tuple[str, str, str]] = []  # (room_a, room_b, connection_type)

        # Object locations
        self._object_locations: dict[str, dict[str, Any]] = {}

        # Navigable area (2D grid for now)
        self._navigable_grid: dict[tuple[int, int], bool] = {}

        # Current robot belief
        self._current_room: str = "unknown"
        self._robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- Room Management ---

    def add_room(self, room_id: str, name: str, bounds: dict[str, Any] | None = None) -> None:
        """Register a room in the spatial map."""
        self._rooms[room_id] = {
            "name": name,
            "bounds": bounds or {},
            "objects": [],
        }

    def connect_rooms(self, room_a: str, room_b: str, connection_type: str = "doorway") -> None:
        """Connect two rooms (e.g., via doorway, corridor)."""
        self._room_connections.append((room_a, room_b, connection_type))

    def get_room_location(self, room_name: str) -> dict[str, Any] | None:
        """Find a room by name and return its info."""
        for room_id, info in self._rooms.items():
            if info["name"].lower() == room_name.lower():
                return {"room_id": room_id, **info}
        return None

    def get_connected_rooms(self, room_id: str) -> list[str]:
        """Get rooms directly connected to the given room."""
        connected = []
        for a, b, _ in self._room_connections:
            if a == room_id:
                connected.append(b)
            elif b == room_id:
                connected.append(a)
        return connected

    # --- Object Management ---

    def place_object(self, object_id: str, label: str, position: tuple[float, float, float], room_id: str | None = None) -> None:
        """Record an object's location."""
        self._object_locations[object_id] = {
            "label": label,
            "position": position,
            "room_id": room_id,
        }
        if room_id and room_id in self._rooms:
            self._rooms[room_id]["objects"].append(object_id)

    def get_objects_in_room(self, room_id: str) -> list[dict[str, Any]]:
        """Get all objects believed to be in a room."""
        obj_ids = self._rooms.get(room_id, {}).get("objects", [])
        return [
            {"object_id": oid, **self._object_locations[oid]}
            for oid in obj_ids
            if oid in self._object_locations
        ]

    def find_object(self, label: str) -> dict[str, Any] | None:
        """Find an object by label."""
        for obj_id, info in self._object_locations.items():
            if info["label"].lower() == label.lower():
                return {"object_id": obj_id, **info}
        return None

    # --- Navigation ---

    def set_navigable(self, x_idx: int, y_idx: int, navigable: bool) -> None:
        """Mark a grid cell as navigable or obstructed."""
        self._navigable_grid[(x_idx, y_idx)] = navigable

    def is_navigable(self, position: tuple[float, float]) -> bool:
        """Check if a world position is in a navigable area."""
        x_idx = int(position[0] / self.resolution_m)
        y_idx = int(position[1] / self.resolution_m)
        return self._navigable_grid.get((x_idx, y_idx), True)

    # --- Robot State ---

    def update_robot_position(self, position: tuple[float, float, float], room_id: str | None = None) -> None:
        """Update the robot's believed position."""
        self._robot_position = position
        if room_id:
            self._current_room = room_id

    @property
    def current_room(self) -> str:
        return self._current_room

    @property
    def robot_position(self) -> tuple[float, float, float]:
        return self._robot_position

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "rooms": self._rooms,
            "connections": self._room_connections,
            "object_locations": self._object_locations,
            "current_room": self._current_room,
            "robot_position": self._robot_position,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        self._rooms = data.get("rooms", {})
        self._room_connections = data.get("connections", [])
        self._object_locations = data.get("object_locations", {})
        self._current_room = data.get("current_room", "unknown")
        self._robot_position = tuple(data.get("robot_position", (0.0, 0.0, 0.0)))
