"""LangChain tools wrapping perception queries for LLM agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_perception_tools(perception_module: Any = None) -> list:
    """Create LangChain tools for perception operations.

    Tools provided:
    - detect_objects: List all objects currently visible
    - look_at: Orient camera toward a target
    - describe_scene: Get a natural language description of the current scene

    Returns:
        List of LangChain StructuredTool objects.
    """
    tools = []

    try:
        from langchain_core.tools import tool

        @tool
        async def detect_objects(category: str = "all") -> str:
            """Detect objects in the current scene. Optionally filter by category (e.g., 'food', 'furniture')."""
            logger.info(f"Tool: detect_objects(category={category})")
            # In production, calls perception module
            return f"Detected objects in category '{category}': [table, cup, chair, door]"

        @tool
        async def look_at(target: str) -> str:
            """Orient the robot's head camera to look at a specific target or direction."""
            logger.info(f"Tool: look_at(target={target})")
            return f"Looking at '{target}'"

        @tool
        async def describe_scene() -> str:
            """Get a natural language description of everything visible in the current scene."""
            logger.info("Tool: describe_scene()")
            return "I see a kitchen with a table, two chairs, a cup on the table, and a doorway to the living room."

        tools = [detect_objects, look_at, describe_scene]
    except ImportError:
        logger.warning("LangChain not available; perception tools not created")

    return tools
