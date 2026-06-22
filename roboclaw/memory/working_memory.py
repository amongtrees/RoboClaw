"""Working memory — short-term state cache backed by Redis with in-process fallback.

Stores the latest RobotState, a rolling buffer of recent events, and task context.
Redis keys are namespaced: roboclaw:{robot_id}:working_memory
"""

from __future__ import annotations

import json
from collections import deque
from time import time
from typing import Any

from roboclaw.models.state import HumanoidState, RobotState


class WorkingMemory:
    """Short-term memory for current task context and robot state.

    Primary backend: Redis Hash (with TTL for auto-expiry).
    Fallback: in-process dict (for development without Redis).
    """

    def __init__(
        self,
        robot_id: str,
        redis_client: Any = None,
        max_events: int = 200,
        state_ttl_s: int = 10,
    ) -> None:
        self.robot_id = robot_id
        self._redis = redis_client
        self._max_events = max_events
        self._state_ttl_s = state_ttl_s

        # In-process fallback
        self._state: dict[str, Any] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._task_context: dict[str, Any] = {}

    @property
    def _key(self) -> str:
        return f"roboclaw:{self.robot_id}:working_memory"

    @property
    def _events_key(self) -> str:
        return f"roboclaw:{self.robot_id}:events"

    # --- State ---

    async def update_state(self, **kwargs: Any) -> None:
        """Update working memory state fields."""
        self._state.update(kwargs)
        self._state["timestamp"] = time()
        if self._redis:
            try:
                serializable = {k: json.dumps(v) if not isinstance(v, (str, int, float)) else v for k, v in self._state.items()}
                await self._redis.hset(self._key, mapping=serializable)
                await self._redis.expire(self._key, self._state_ttl_s)
            except Exception:
                pass  # graceful degradation to in-process

    async def get_state(self) -> dict[str, Any]:
        """Get the current working memory state."""
        if self._redis:
            try:
                raw = await self._redis.hgetall(self._key)
                if raw:
                    state: dict[str, Any] = {}
                    for k, v in raw.items():
                        k_str = k.decode() if isinstance(k, bytes) else k
                        v_str = v.decode() if isinstance(v, bytes) else v
                        try:
                            state[k_str] = json.loads(v_str)
                        except (json.JSONDecodeError, TypeError):
                            state[k_str] = v_str
                    return state
            except Exception:
                pass
        return dict(self._state)

    async def get_robot_state(self) -> RobotState:
        """Reconstruct a RobotState from working memory."""
        state = await self.get_state()
        return HumanoidState(
            robot_id=self.robot_id,
            timestamp=state.get("timestamp", time()),
            phase=state.get("phase"),
            task_id=state.get("task_id"),
            skill_name=state.get("skill_name"),
            joint_states=state.get("joint_positions", {}),
            visible_objects=state.get("detected_objects", []),
            safety_state=state.get("safety_status", {}),
            zmp=tuple(state.get("zmp", (0.0, 0.0))),
        )

    # --- Events ---

    async def add_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Add an event to the rolling buffer."""
        event = {
            "timestamp": time(),
            "event_type": event_type,
            "payload": payload or {},
        }
        self._events.append(event)
        if self._redis:
            try:
                await self._redis.lpush(self._events_key, json.dumps(event))
                await self._redis.ltrim(self._events_key, 0, self._max_events - 1)
            except Exception:
                pass

    async def get_recent_events(self, n: int = 50) -> list[dict[str, Any]]:
        """Get the most recent N events."""
        if self._redis:
            try:
                raw = await self._redis.lrange(self._events_key, 0, n - 1)
                return [json.loads(e.decode() if isinstance(e, bytes) else e) for e in raw]
            except Exception:
                pass
        return list(self._events)[-n:]

    # --- Task Context ---

    async def set_task_context(self, task_id: str, context: dict[str, Any]) -> None:
        """Store task-specific context."""
        self._task_context = {"task_id": task_id, **context}
        await self.update_state(task_id=task_id)

    async def get_task_context(self) -> dict[str, Any]:
        return dict(self._task_context)

    async def clear_task(self) -> None:
        """Clear task-specific context."""
        self._task_context.clear()
        await self.update_state(task_id=None, skill_name=None)

    # --- Lifecycle ---

    async def close(self) -> None:
        """Clean up resources."""
        if self._redis:
            try:
                await self._redis.delete(self._key, self._events_key)
            except Exception:
                pass
