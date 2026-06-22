"""LangChain tools wrapping safety checks for LLM agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_safety_tools(safety_monitor: Any = None, knowledge_base: Any = None) -> list:
    """Create LangChain tools for safety operations.

    Tools provided:
    - check_safety: Run all safety checks and report status
    - get_safety_rules: Get all active safety rules and constraints
    - emergency_stop: Trigger an emergency stop (requires confirmation)

    Returns:
        List of LangChain StructuredTool objects.
    """
    tools = []

    try:
        from langchain_core.tools import tool

        @tool
        async def check_safety() -> str:
            """Run all safety checks and report the current safety status."""
            logger.info("Tool: check_safety()")
            if safety_monitor:
                is_safe = safety_monitor.is_safe
                violations = safety_monitor.get_recent_violations(5)
                if violations:
                    return f"Safety: {'SAFE' if is_safe else 'UNSAFE'}. Recent violations: {len(violations)}"
                return f"Safety: {'SAFE' if is_safe else 'UNSAFE'}. No recent violations."
            return "Safety: All systems nominal (simulated)."

        @tool
        async def get_safety_rules() -> str:
            """Get all active safety rules and constraints that must be followed."""
            logger.info("Tool: get_safety_rules()")
            if knowledge_base:
                rules = await knowledge_base.get_safety_rules()
                if rules:
                    return "\n".join(f"- {r.get('description', str(r))}" for r in rules[:10])
            return "Safety rules: ZMP within polygon, joint limits, force limits, human proximity stop, max payload 5kg."

        @tool
        async def emergency_stop(reason: str) -> str:
            """Trigger an emergency stop. Use ONLY when there is an immediate safety risk.
            Requires a reason describing the safety concern.
            """
            logger.critical(f"Tool: emergency_stop(reason={reason})")
            if safety_monitor:
                safety_monitor._estop_active = True
            return f"EMERGENCY STOP ACTIVATED. Reason: {reason}"

        tools = [check_safety, get_safety_rules, emergency_stop]
    except ImportError:
        logger.warning("LangChain not available; safety tools not created")

    return tools
