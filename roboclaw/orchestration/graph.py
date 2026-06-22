"""Main LangGraph StateGraph definition — the central nervous system of RoboClaw.

Compiles the full agent loop: PERCEIVE → REFLECT → PLAN → ACT → FEEDBACK
with conditional routing for recovery, delegation, and HRI.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from roboclaw.orchestration.checkpointer import create_memory_checkpointer
from roboclaw.orchestration.edges import (
    route_after_feedback,
    route_after_hri,
    route_after_recovery,
)
from roboclaw.orchestration.nodes import (
    act_node,
    delegate_a2a_node,
    feedback_node,
    hri_node,
    perceive_node,
    plan_node,
    recover_node,
    reflect_node,
)
from roboclaw.orchestration.state_schema import AgentGraphState

logger = logging.getLogger(__name__)


def build_agent_graph(checkpointer: Any = None) -> Any:
    """Build and compile the RoboClaw agent StateGraph.

    The graph implements the main agent loop:
        PERCEIVE → REFLECT → PLAN → ACT → FEEDBACK
            ↑                                  ↓
            └──────── (continue) ──────────────┘
                          │
                    (recover / delegate / hri / complete)

    Args:
        checkpointer: LangGraph checkpointer (PostgresSaver or MemorySaver).
                      If None, uses in-memory checkpointer.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    if checkpointer is None:
        checkpointer = create_memory_checkpointer()

    graph = StateGraph(AgentGraphState)

    # --- Add nodes ---
    graph.add_node("perceive", perceive_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("recover", recover_node)
    graph.add_node("delegate_a2a", delegate_a2a_node)
    graph.add_node("handle_hri", hri_node)

    # --- Set entry point ---
    graph.set_entry_point("perceive")

    # --- Core loop edges ---
    graph.add_edge("perceive", "reflect")
    graph.add_edge("reflect", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "feedback")

    # --- Conditional routing from feedback ---
    graph.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "continue": "perceive",
            "recover": "recover",
            "delegate": "delegate_a2a",
            "hri": "handle_hri",
            "complete": END,
        },
    )

    # --- Conditional routing from recovery ---
    graph.add_conditional_edges(
        "recover",
        route_after_recovery,
        {
            "replan": "plan",
            "retry": "act",
            "abort": END,
            "human_help": "handle_hri",
        },
    )

    # --- Post-HRI and post-delegation ---
    graph.add_edge("handle_hri", "perceive")
    graph.add_edge("delegate_a2a", "feedback")

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("RoboClaw agent graph compiled successfully")
    return compiled


async def create_agent_runner(
    robot_id: str,
    checkpointer: Any = None,
) -> Any:
    """Create a ready-to-run agent graph with default configuration.

    Args:
        robot_id: The robot's unique identifier.
        checkpointer: Optional checkpointer (uses in-memory if None).

    Returns:
        Tuple of (compiled_graph, config_dict) ready for graph.ainvoke().
    """
    graph = build_agent_graph(checkpointer)

    config = {
        "configurable": {
            "thread_id": f"roboclaw-{robot_id}",
            "checkpoint_ns": robot_id,
        }
    }

    return graph, config
