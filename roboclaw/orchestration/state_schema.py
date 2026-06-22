"""LangGraph State schema — the full state flowing through the agent graph."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


def _new_session_id() -> str:
    return f"session_{uuid4().hex[:12]}"


class PerceptionData(BaseModel):
    """Aggregated perception data for one cycle (stub — real data flows via memory)."""

    timestamp: float = 0.0
    detected_objects: list[dict[str, Any]] = Field(default_factory=list)
    speech_text: str | None = None
    sound_events: list[str] = Field(default_factory=list)
    scene_graph: dict[str, Any] = Field(default_factory=dict)


class AgentGraphState(BaseModel):
    """The full state flowing through the LangGraph StateGraph.

    This is the single source of truth for the agent during execution.
    It is checkpointed by LangGraph's PostgresSaver for fault tolerance.
    """

    # Identity
    robot_id: str = ""
    session_id: str = Field(default_factory=_new_session_id)

    # Task
    task_spec: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    current_subtask: dict[str, Any] | None = None
    completed_subtask_ids: list[str] = Field(default_factory=list)
    task_history: list[dict[str, Any]] = Field(default_factory=list)

    # Perception (latest cycle)
    perception: PerceptionData | None = None
    latest_state_summary: str = ""

    # World state
    robot_pose: dict[str, Any] | None = None
    world_objects: list[dict[str, Any]] = Field(default_factory=list)
    safety_status: dict[str, Any] = Field(default_factory=dict)

    # Memory pointers
    episode_id: str | None = None
    relevant_memories: list[dict[str, Any]] = Field(default_factory=list)

    # Conversation / HRI (messages managed by LangGraph's add_messages reducer)
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # Control flow
    loop_count: int = 0
    phase: str = "idle"
    error: dict[str, Any] | None = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    should_continue: bool = True
    should_delegate: bool = False
    require_human_help: bool = False

    # A2A (for multi-agent scenarios)
    peer_agents: list[dict[str, Any]] = Field(default_factory=list)
    pending_delegations: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
