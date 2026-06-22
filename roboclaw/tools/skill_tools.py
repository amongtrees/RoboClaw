"""LangChain tools wrapping skill invocation for LLM agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_skill_tools(skill_library: Any = None) -> list:
    """Create LangChain tools for skill execution.

    Tools provided:
    - execute_skill: Execute a named robot skill with parameters
    - list_available_skills: List all skills the robot can perform
    - get_skill_info: Get detailed information about a specific skill

    Returns:
        List of LangChain StructuredTool objects.
    """
    tools = []

    try:
        from langchain_core.tools import tool

        @tool
        async def execute_skill(skill_name: str, parameters_json: str = "{}") -> str:
            """Execute a robot skill by name. Provide parameters as a JSON string.
            Example: execute_skill('whole_body_grasp', '{"object": "cup", "arm": "right"}')
            """
            import json

            logger.info(f"Tool: execute_skill(name={skill_name}, params={parameters_json})")
            try:
                params = json.loads(parameters_json)
            except json.JSONDecodeError:
                params = {}

            if skill_library:
                result = await skill_library.execute_skill(skill_name, params)
                return f"Skill '{skill_name}': {result.status} — {result.actual_outcome}"
            return f"Skill '{skill_name}' executed successfully (simulated)."

        @tool
        async def list_available_skills() -> str:
            """List all skills the robot can perform."""
            logger.info("Tool: list_available_skills()")
            if skill_library:
                skills = skill_library.list_skill_names()
                return f"Available skills: {', '.join(skills)}"
            return "Skills: navigate_to, walk_steps, whole_body_grasp, place_object, handover, speak, gaze_at, open_door, climb_stairs"

        @tool
        async def get_skill_info(skill_name: str) -> str:
            """Get detailed information about a specific skill, including its parameters."""
            logger.info(f"Tool: get_skill_info(name={skill_name})")
            if skill_library:
                definition = skill_library.get_definition(skill_name)
                if definition:
                    params = ", ".join(
                        f"{p.name}: {p.param_type}" + (" (required)" if p.required else " (optional)")
                        for p in definition.parameters
                    )
                    return f"{definition.name}: {definition.description}\nParameters: {params}\nTimeout: {definition.timeout_sec}s"
            return f"Skill '{skill_name}': A humanoid robot skill."

        tools = [execute_skill, list_available_skills, get_skill_info]
    except ImportError:
        logger.warning("LangChain not available; skill tools not created")

    return tools
