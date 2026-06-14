"""TTL-backed working memory for the robot's current state."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import RLock
from time import time
from typing import Any

from .models import ForceTorque, Pose, RobotState


class WorkingMemory:
    """Stores the current robot state and a small rolling event buffer."""

    def __init__(
        self,
        robot_id: str,
        ttl_sec: float = 5.0,
        max_recent_events: int = 100,
    ) -> None:
        self.robot_id = robot_id
        self.ttl_sec = ttl_sec
        self._lock = RLock()
        self._state = RobotState(robot_id=robot_id)
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=max_recent_events)

    def update_state(self, **updates: Any) -> RobotState:
        with self._lock:
            state_data = self._state.to_dict()
            state_data.update(updates)
            state_data["robot_id"] = self.robot_id
            state_data["timestamp"] = float(updates.get("timestamp", time()))

            if isinstance(state_data.get("end_effector_pose"), dict):
                state_data["end_effector_pose"] = Pose.from_dict(
                    state_data["end_effector_pose"]
                )
            if isinstance(state_data.get("force_torque"), dict):
                state_data["force_torque"] = ForceTorque.from_dict(
                    state_data["force_torque"]
                )

            self._state = RobotState(**state_data)
            self._recent_events.append(
                {
                    "timestamp": self._state.timestamp,
                    "event_type": "state_updated",
                    "payload": deepcopy(updates),
                }
            )
            return deepcopy(self._state)

    def add_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._recent_events.append(
                {
                    "timestamp": time(),
                    "event_type": event_type,
                    "payload": deepcopy(payload or {}),
                }
            )

    def get_state(self, include_staleness: bool = True) -> dict[str, Any]:
        with self._lock:
            data = self._state.to_dict()
            if include_staleness:
                age_sec = max(0.0, time() - self._state.timestamp)
                data["age_sec"] = age_sec
                data["is_stale"] = age_sec > self.ttl_sec
            data["recent_events"] = list(self._recent_events)
            return deepcopy(data)

    def clear(self) -> None:
        with self._lock:
            self._state = RobotState(robot_id=self.robot_id)
            self._recent_events.clear()
