"""Reusable LangGraph sub-graphs for common robot task sequences.

Sub-graphs encapsulate multi-step behaviors that can be composed
into the main agent graph or called from other sub-graphs.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from roboclaw.orchestration.state_schema import AgentGraphState

logger = logging.getLogger(__name__)


# --- Grasp Sequence Sub-Graph ---


async def _grasp_approach(state: AgentGraphState) -> dict[str, Any]:
    """Approach the target object."""
    logger.debug("Sub-graph: grasp_approach")
    return {"phase": "grasp_approach"}


async def _grasp_execute(state: AgentGraphState) -> dict[str, Any]:
    """Execute the grasp."""
    logger.debug("Sub-graph: grasp_execute")
    return {"phase": "grasp_execute"}


async def _grasp_verify(state: AgentGraphState) -> dict[str, Any]:
    """Verify grasp stability."""
    logger.debug("Sub-graph: grasp_verify")
    return {"phase": "grasp_verify"}


def _route_grasp(state: AgentGraphState) -> str:
    """Route within grasp sub-graph."""
    if state.error:
        return "retry"
    return "complete"


def build_grasp_subgraph() -> StateGraph:
    """Build a reusable sub-graph for whole-body grasp sequence."""
    graph = StateGraph(AgentGraphState)

    graph.add_node("grasp_approach", _grasp_approach)
    graph.add_node("grasp_execute", _grasp_execute)
    graph.add_node("grasp_verify", _grasp_verify)

    graph.set_entry_point("grasp_approach")
    graph.add_edge("grasp_approach", "grasp_execute")
    graph.add_edge("grasp_execute", "grasp_verify")
    graph.add_conditional_edges("grasp_verify", _route_grasp, {
        "retry": "grasp_approach",
        "complete": END,
    })

    return graph.compile()


# --- Walk-to-Pose Sub-Graph ---


async def _walk_plan_steps(state: AgentGraphState) -> dict[str, Any]:
    """Plan footstep sequence to target pose."""
    logger.debug("Sub-graph: walk_plan_steps")
    return {"phase": "walk_plan"}


async def _walk_execute_step(state: AgentGraphState) -> dict[str, Any]:
    """Execute one walking step."""
    logger.debug("Sub-graph: walk_execute_step")
    return {"phase": "walk_execute"}


async def _walk_check_balance(state: AgentGraphState) -> dict[str, Any]:
    """Check balance after step."""
    logger.debug("Sub-graph: walk_check_balance")
    return {"phase": "walk_balance_check"}


def build_walk_subgraph() -> StateGraph:
    """Build a reusable sub-graph for bipedal walking sequence."""
    graph = StateGraph(AgentGraphState)

    graph.add_node("walk_plan", _walk_plan_steps)
    graph.add_node("walk_execute", _walk_execute_step)
    graph.add_node("walk_balance", _walk_check_balance)

    graph.set_entry_point("walk_plan")
    graph.add_edge("walk_plan", "walk_execute")
    graph.add_edge("walk_execute", "walk_balance")
    graph.add_conditional_edges("walk_balance", lambda s: "complete" if not s.error else "retry", {
        "retry": "walk_plan",
        "complete": END,
    })

    return graph.compile()
