"""Conditional edge routing logic for the LangGraph agent loop.

Each routing function examines the current state and decides which node
to execute next. These are the decision points in the agent's control flow.
"""

from __future__ import annotations

import logging

from roboclaw.orchestration.state_schema import AgentGraphState

logger = logging.getLogger(__name__)


def route_after_feedback(state: AgentGraphState) -> str:
    """Decide next step after evaluating sub-task outcome.

    Returns one of: continue, recover, delegate, hri, complete
    """
    # Check if we need to stop
    if not state.should_continue:
        logger.info(f"[{state.robot_id}] Agent loop complete — all tasks finished")
        return "complete"

    # Error recovery
    if state.error:
        if state.recovery_attempts < state.max_recovery_attempts:
            logger.info(f"[{state.robot_id}] Error detected, routing to recovery (attempt {state.recovery_attempts + 1})")
            return "recover"
        if state.require_human_help:
            logger.info(f"[{state.robot_id}] Recovery exhausted, routing to HRI")
            return "hri"
        logger.warning(f"[{state.robot_id}] Unrecoverable error, aborting")
        return "complete"

    # Delegation
    if state.should_delegate and state.peer_agents:
        logger.info(f"[{state.robot_id}] Delegating to peer agent")
        return "delegate"

    # Continue the loop
    return "continue"


def route_after_recovery(state: AgentGraphState) -> str:
    """Decide next step after recovery attempt.

    Returns one of: replan, retry, abort, human_help
    """
    if state.error is None:
        # Recovery succeeded — retry the failed sub-task
        logger.info(f"[{state.robot_id}] Recovery succeeded, retrying action")
        return "retry"

    if state.require_human_help:
        logger.info(f"[{state.robot_id}] Recovery failed, requesting human help")
        return "human_help"

    # Need to re-plan with updated world state
    error = state.error or {}
    if error.get("type") in ("planning", "kinematic"):
        logger.info(f"[{state.robot_id}] Re-planning after {error.get('type')} error")
        return "replan"

    # Default: abort
    logger.error(f"[{state.robot_id}] Recovery failed, aborting")
    return "abort"


def route_after_hri(state: AgentGraphState) -> str:
    """Decide next step after human-robot interaction.

    Returns: continue (back to perceive) or abort.
    """
    # If operator resolved the issue, go back to perceiving
    if state.error is None:
        return "continue"
    return "abort"
