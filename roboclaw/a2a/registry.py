"""Agent registry — local directory of known peer agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Maintains a directory of known peer agents for multi-agent collaboration.

    Supports static configuration and dynamic discovery.
    """

    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}

    def register(self, agent_name: str, agent_card: dict[str, Any], url: str) -> None:
        """Register a peer agent."""
        self._agents[agent_name] = {
            "name": agent_name,
            "agent_card": agent_card,
            "url": url,
            "status": "available",
            "last_seen": None,
        }
        logger.info(f"Registered peer agent: {agent_name} at {url}")

    def unregister(self, agent_name: str) -> None:
        """Remove a peer agent."""
        self._agents.pop(agent_name, None)

    def get_agent(self, agent_name: str) -> dict[str, Any] | None:
        """Get a registered agent by name."""
        return self._agents.get(agent_name)

    def find_by_capability(self, capability_id: str) -> list[dict[str, Any]]:
        """Find agents that have a specific capability."""
        results = []
        for agent in self._agents.values():
            capabilities = agent.get("agent_card", {}).get("capabilities", [])
            if any(c.get("capability_id") == capability_id for c in capabilities):
                results.append(agent)
        return results

    def list_available(self) -> list[dict[str, Any]]:
        """List all available agents."""
        return [a for a in self._agents.values() if a["status"] == "available"]

    def mark_unavailable(self, agent_name: str) -> None:
        """Mark an agent as unavailable."""
        if agent_name in self._agents:
            self._agents[agent_name]["status"] = "unavailable"

    def mark_available(self, agent_name: str) -> None:
        """Mark an agent as available."""
        if agent_name in self._agents:
            self._agents[agent_name]["status"] = "available"

    def __len__(self) -> int:
        return len(self._agents)
