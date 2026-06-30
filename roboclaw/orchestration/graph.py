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
    model_router: Any = None,
    sim_module: Any = None,
    ros2_module: Any = None,
    llm: Any = None,
) -> Any:
    """Create a ready-to-run agent graph with default configuration.

    Args:
        robot_id: The robot's unique identifier.
        checkpointer: Optional checkpointer (uses in-memory if None).
        model_router: Optional ModelRouter for VLN/VLA/World Model dispatch.
                      When None, a default ModelRouter is created with optional
                      sim executor wiring.
        sim_module: Optional SimModule.  When provided, MuJoCo physics executors
                    are registered on the SkillLibrary used by the ModelRouter.
        ros2_module: Optional ROS2Module.  When provided, ROS 2 executors are
                     registered on the SkillLibrary.  ROS2Module takes priority
                     over SimModule for the same skill types.

    Returns:
        Tuple of (compiled_graph, config_dict) ready for graph.ainvoke().
    """
    # --- Create SkillLibrary and wire executors ---
    from roboclaw.action.skill_library import SkillLibrary

    skill_library = SkillLibrary()

    # MuJoCo sim executors (physics simulation fallback)
    if sim_module is not None:
        sim_module.wire_executors(skill_library)
        logger.info("Simulation executors wired into SkillLibrary")

    # ROS 2 executors (real hardware — registers after sim, takes priority
    # because SkillLibrary.register_executor overwrites previous registration)
    if ros2_module is not None:
        ros2_module.wire_executors(skill_library)
        logger.info("ROS 2 executors wired into SkillLibrary")

    # --- Install ModelRouter ---
    if model_router is None:
        # Create a default router that uses the (optionally sim-wired) SkillLibrary
        from roboclaw.clients.model_router import ModelRouter

        model_router = ModelRouter(skill_library=skill_library)
    elif getattr(model_router, '_skill_library', None) is None:
        # Router was pre-created without a SkillLibrary — attach ours
        model_router._skill_library = skill_library

    from roboclaw.orchestration.nodes import set_model_router
    set_model_router(model_router)

    # --- Install SensorFusion for perception pipeline ---
    if sim_module is not None:
        from roboclaw.perception.sensor_fusion import SensorFusion
        from roboclaw.orchestration.nodes import set_sensor_fusion

        sf = SensorFusion(env=sim_module.env)
        set_sensor_fusion(sf)
        logger.info("SensorFusion installed for perception pipeline (MuJoCo backend)")
    else:
        logger.info("No SimModule — perception will use stubs")

    # --- Install LLM for TaskDecomposer ---
    if llm is not None:
        from roboclaw.orchestration.nodes import set_task_decomposer_llm
        set_task_decomposer_llm(llm)
        logger.info("LLM installed for TaskDecomposer planning")
    else:
        logger.info("No LLM — TaskDecomposer will use template-based planning")

    graph = build_agent_graph(checkpointer)

    config = {
        "configurable": {
            "thread_id": f"roboclaw-{robot_id}",
            "checkpoint_ns": robot_id,
        }
    }

    return graph, config
