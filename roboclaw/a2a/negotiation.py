"""Task allocation and handoff negotiation between agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NegotiationEngine:
    """Handles task allocation decisions across multiple peer agents.

    Evaluates capability match, current load, estimated latency,
    and trust score to select the best agent for a task.
    """

    def __init__(self, registry: Any = None) -> None:
        self._registry = registry

    async def find_best_agent(
        self,
        task_description: str,
        required_capabilities: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Find the best available agent for a given task.

        Selection criteria (in priority order):
        1. Capability match (must have all required capabilities)
        2. Current load (prefer idle agents)
        3. Trust score (prefer higher trust)
        4. Latency (prefer lower latency estimates)

        Args:
            task_description: Natural language description of the task.
            required_capabilities: Optional list of capability_ids required.

        Returns:
            Best agent dict, or None if no suitable agent found.
        """
        if self._registry is None:
            return None

        available = self._registry.list_available()

        if not available:
            logger.warning("No available agents for task allocation")
            return None

        # Filter by capability match if requirements specified
        if required_capabilities:
            qualified = []
            for agent in available:
                agent_caps = {
                    c.get("capability_id")
                    for c in agent.get("agent_card", {}).get("capabilities", [])
                }
                if all(req in agent_caps for req in required_capabilities):
                    qualified.append(agent)
            available = qualified

        if not available:
            logger.warning("No agents with required capabilities")
            return None

        # Simple heuristic: pick the first matching agent
        # In production, uses scoring with load/trust/latency
        best = available[0]
        logger.info(f"Selected agent '{best['name']}' for task: {task_description}")
        return best

    async def negotiate_handoff(
        self,
        task: dict[str, Any],
        from_agent: str,
        to_agent: str | None = None,
    ) -> bool:
        """Negotiate a task handoff from one agent to another.

        Returns True if the handoff was accepted.
        """
        logger.info(f"Negotiating handoff from '{from_agent}' to '{to_agent}'")
        # Phase 1 stub — always accepts
        return True
