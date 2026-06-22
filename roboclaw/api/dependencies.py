"""FastAPI dependency injection for agent runtime components."""

from __future__ import annotations

from typing import Any


async def get_agent_runtime():
    """Dependency: get the agent runtime (graph runner).

    In production, this returns a singleton agent runner instance.
    For Phase 1, returns a stub.
    """
    # Stub — in production, returns the compiled LangGraph runner
    return None


async def get_episodic_memory():
    """Dependency: get the episodic memory store."""
    return None


async def get_skill_library():
    """Dependency: get the skill library."""
    return None
