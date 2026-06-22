"""LangChain tools wrapping knowledge base queries for LLM agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_knowledge_tools(knowledge_base: Any = None) -> list:
    """Create LangChain tools for knowledge base operations.

    Tools provided:
    - query_affordance: Get manipulation affordances for an object
    - query_knowledge: Query the knowledge base for domain facts

    Returns:
        List of LangChain StructuredTool objects.
    """
    tools = []

    try:
        from langchain_core.tools import tool

        @tool
        async def query_affordance(object_label: str) -> str:
            """Get manipulation affordances for an object type (how to grasp it, required force, etc.)."""
            logger.info(f"Tool: query_affordance(object={object_label})")
            if knowledge_base:
                affordance = await knowledge_base.get_affordance(object_label)
                return f"Affordance for '{object_label}': grasp_type={affordance.get('grasp_type', 'unknown')}, grip_force={affordance.get('grip_force_n', '?')}N"
            return f"No affordance data for '{object_label}'."

        @tool
        async def query_knowledge(domain: str, key: str) -> str:
            """Query the robot's knowledge base for domain facts. Domains: object_affordance, safety_rule, kinematic_limit, skill_parameter."""
            logger.info(f"Tool: query_knowledge(domain={domain}, key={key})")
            if knowledge_base:
                result = await knowledge_base.query(domain, key)
                if result:
                    return f"{domain}/{key}: {result}"
            return f"No knowledge found for {domain}/{key}."

        tools = [query_affordance, query_knowledge]
    except ImportError:
        logger.warning("LangChain not available; knowledge tools not created")

    return tools
