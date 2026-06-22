"""Global and local navigation planner stub.

In production, integrates with A*, RRT-Connect, or DWA planners
using the robot's spatial memory and occupancy grid.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NavigationPlanner:
    """Global path planner for humanoid navigation.

    Phase 1 stub — uses straight-line planning via spatial memory.
    """

    def __init__(self, spatial_memory: Any = None) -> None:
        self._spatial = spatial_memory

    async def plan_path(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        """Plan a collision-free path from start to goal.

        Phase 1: returns straight-line waypoints.
        Production: A* on costmap, RRT-Connect for 3D, DWA for local.
        """
        # Simple straight-line with intermediate waypoints
        distance = (
            (goal[0] - start[0]) ** 2
            + (goal[1] - start[1]) ** 2
            + (goal[2] - start[2]) ** 2
        ) ** 0.5

        num_waypoints = max(2, int(distance / 0.5))  # waypoint every 0.5m
        path = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            path.append((
                start[0] + (goal[0] - start[0]) * t,
                start[1] + (goal[1] - start[1]) * t,
                start[2] + (goal[2] - start[2]) * t,
            ))

        logger.info(f"Planned path: {len(path)} waypoints over {distance:.2f}m")
        return path

    async def plan_room_navigation(
        self,
        current_room: str,
        target_room: str,
    ) -> list[str]:
        """Plan a sequence of rooms to traverse from current to target.

        Uses the spatial memory room graph for pathfinding.
        """
        if self._spatial is None:
            return [current_room, target_room]

        # BFS through room graph
        visited = {current_room}
        queue = [(current_room, [current_room])]

        while queue:
            room, path = queue.pop(0)
            if room == target_room:
                logger.info(f"Room path: {' → '.join(path)}")
                return path

            for neighbor in self._spatial.get_connected_rooms(room):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        logger.warning(f"No path found from '{current_room}' to '{target_room}'")
        return [current_room, target_room]
