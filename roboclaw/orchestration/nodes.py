"""LangGraph node implementations — thin orchestrators that delegate to sub-modules.

Each node is a pure function: (state) -> dict of state updates.
This keeps the graph definition clean and each module independently testable.
"""

from __future__ import annotations

import logging
from time import time
from typing import Any

from roboclaw.core.types import AgentPhase
from roboclaw.orchestration.state_schema import AgentGraphState, PerceptionData

logger = logging.getLogger(__name__)


async def perceive_node(state: AgentGraphState) -> dict[str, Any]:
    """Ingest sensor data and produce a unified perception snapshot.

    When a ``SensorFusion`` singleton is available (simulation or hardware
    mode), delegates to it for real multi-modal sensor fusion.  Falls back
    to stub perception data otherwise.
    """
    logger.debug(f"[{state.robot_id}] PERCEIVE phase — loop {state.loop_count}")

    sf = _get_sensor_fusion()

    if sf is not None:
        try:
            result = await sf.fuse(state)
            result["phase"] = AgentPhase.PERCEIVING
            result["loop_count"] = state.loop_count + 1
            return result
        except Exception:
            logger.exception("SensorFusion.fuse() failed, falling back to stub")

    # Fallback stub — no simulation or SensorFusion not configured
    perception = PerceptionData(
        timestamp=time(),
        detected_objects=state.world_objects,
    )

    return {
        "perception": perception,
        "phase": AgentPhase.PERCEIVING,
        "loop_count": state.loop_count + 1,
    }


async def reflect_node(state: AgentGraphState) -> dict[str, Any]:
    """Update working memory, check safety constraints, log significant changes.

    This is a lightweight node that runs every cycle.
    """
    logger.debug(f"[{state.robot_id}] REFLECT phase")

    # Compose a state summary from perception
    perception = state.perception
    objects_seen = len(perception.detected_objects) if perception else 0

    # Build richer summary when sensor data is available
    summary_parts = [f"Objects: {objects_seen}"]
    com = state.com_position
    zmp = state.zmp_position
    contacts = state.contact_forces
    if com and len(com) >= 3:
        summary_parts.append(f"COM: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
    if zmp and len(zmp) >= 2:
        summary_parts.append(f"ZMP: [{zmp[0]:.3f}, {zmp[1]:.3f}]")
    if contacts:
        summary_parts.append(f"Contacts: {len(contacts)}")
    summary_parts.append(f"Phase: {state.phase}")

    summary = " | ".join(summary_parts)

    # Replace the simple summary with the rich one when sensor data exists
    if not (com or zmp or contacts):
        summary = f"Objects detected: {objects_seen}. Phase: {state.phase}."

    # Check safety (in production, calls SafetyMonitor)
    safety_ok = state.safety_status.get("estop", False) is False

    return {
        "latest_state_summary": summary,
        "phase": AgentPhase.REFLECTING,
        "should_continue": safety_ok,
    }


async def plan_node(state: AgentGraphState) -> dict[str, Any]:
    """Decompose high-level task into executable sub-tasks with model routing.

    Calls TaskDecomposer.plan() which returns (ExecutionPlan, routing_map).
    The routing_map annotates each sub-task with the ModelType that should
    execute it (VLN / VLA / World Model / LLM / NONE).

    When no decomposer is available, generates a stub plan with model tags.
    """
    logger.debug(f"[{state.robot_id}] PLAN phase")

    task_spec = state.task_spec or {}
    goal = task_spec.get("goal", "no goal specified")

    # --- Try TaskDecomposer first ---
    try:
        from roboclaw.planning.task_decomposer import TaskDecomposer
        from roboclaw.planning.plan_validator import PlanValidator

        llm = _get_task_decomposer_llm()
        decomposer = TaskDecomposer(llm=llm, validator=PlanValidator())
        plan, routing_map = await decomposer.plan(task_spec, {}, None)
        return {
            "execution_plan": plan.model_dump(),
            "model_routing": routing_map,
            "phase": AgentPhase.PLANNING,
        }
    except Exception as exc:
        logger.warning("TaskDecomposer not available, using stub plan: %s", exc)

    # --- Stub plan with explicit model routing ---
    plan = {
        "plan_id": f"plan_{state.session_id}",
        "task_goal": goal,
        "sub_tasks": [
            {
                "sub_task_id": "perceive_scene",
                "skill_type": "gaze_at",
                "parameters": {"target": "scene", "_model_type": "llm"},
                "preconditions": [],
                "expected_outcome": {"objects_detected": True},
                "timeout_sec": 5.0,
                "priority": 0,
                "retry_policy": "default",
            },
            {
                "sub_task_id": "navigate_to_target",
                "skill_type": "navigate_to",
                "parameters": {"room": "kitchen", "_model_type": "vln"},
                "preconditions": ["perceive_scene"],
                "expected_outcome": {"at_location": "kitchen"},
                "timeout_sec": 60.0,
                "priority": 1,
                "retry_policy": "default",
            },
            {
                "sub_task_id": "grasp_object",
                "skill_type": "whole_body_grasp",
                "parameters": {"object": "cup", "arm": "right", "_model_type": "vla"},
                "preconditions": ["navigate_to_target"],
                "expected_outcome": {"grasped": True},
                "timeout_sec": 15.0,
                "priority": 2,
                "retry_policy": "default",
            },
        ],
        "estimated_duration_sec": 80.0,
    }

    routing_map = {
        "perceive_scene": "llm",
        "navigate_to_target": "vln",
        "grasp_object": "vla",
    }

    return {
        "execution_plan": plan,
        "model_routing": routing_map,
        "phase": AgentPhase.PLANNING,
    }


async def act_node(state: AgentGraphState) -> dict[str, Any]:
    """Execute the next executable sub-task, routing to VLN/VLA/World Model.

    Uses ModelRouter to dispatch the sub-task to the appropriate model
    client.  When the model is unavailable, falls back to the in-process
    simulated executor via SkillLibrary.
    """
    logger.debug(f"[{state.robot_id}] ACT phase")

    plan = state.execution_plan
    if not plan:
        return {"phase": AgentPhase.ACTING, "error": {"type": "planning", "detail": "No execution plan"}}

    # Find next executable sub-task
    completed = set(state.completed_subtask_ids)
    next_subtask = None
    for st in plan.get("sub_tasks", []):
        if st["sub_task_id"] not in completed and set(st.get("preconditions", [])).issubset(completed):
            next_subtask = st
            break

    if next_subtask:
        sub_task_id = next_subtask["sub_task_id"]
        skill_type = next_subtask.get("skill_type", "")
        model_type = state.model_routing.get(sub_task_id, "llm")

        logger.info(
            f"[{state.robot_id}] Executing sub-task: {sub_task_id} "
            f"({skill_type}) → model={model_type}"
        )

        # --- Try ModelRouter ---
        try:
            from roboclaw.clients.model_router import ModelRouter
            from roboclaw.models.task import SubTask

            router = _get_model_router()
            if router is not None:
                subtask = SubTask(**next_subtask)
                inference = await router.route(
                    subtask,
                    perception=state.perception,
                    world_state={
                        "robot_pose": state.robot_pose,
                        "world_objects": state.world_objects,
                        "safety_status": state.safety_status,
                        "task_goal": plan.get("task_goal", ""),
                    },
                    robot_state={
                        "robot_id": state.robot_id,
                        "joint_positions": state.joint_positions or {},
                        "joint_velocities": state.joint_velocities or {},
                        "imu_data": state.imu_data or {},
                        "com_position": state.com_position or [],
                        "zmp_position": state.zmp_position or [],
                        "body_poses": state.body_poses or {},
                    },
                )
                model_results = dict(state.model_inference_results)
                model_results[sub_task_id] = inference.model_dump()

                # Extract actual execution result from fallback path
                action_result = None
                meta = inference.metadata or {}
                if meta.get("result"):
                    action_result = dict(meta["result"])
                    action_result["status"] = meta.get(
                        "action_status",
                        "success" if inference.is_success else "failure",
                    )
                    action_result["duration_sec"] = meta.get("action_duration_sec", 0)
                    if meta.get("action_error"):
                        action_result["error"] = meta["action_error"]
                elif inference.is_success and router._skill_library is not None:
                    # Model succeeded — still need to drive physics
                    from roboclaw.action.base import ActionResultStatus
                    phys_result = await router._skill_library.execute_skill(
                        skill_type,
                        next_subtask.get("parameters", {}),
                        world_state={
                            "robot_pose": state.robot_pose,
                            "world_objects": state.world_objects,
                            "safety_status": state.safety_status,
                            "task_goal": plan.get("task_goal", ""),
                        },
                        model_result=None,  # force executor path for physics
                    )
                    action_result = phys_result.model_dump()

                return {
                    "current_subtask": next_subtask,
                    "model_inference_results": model_results,
                    "last_action_result": action_result,
                    "phase": AgentPhase.ACTING,
                }
        except Exception as exc:
            logger.warning("ModelRouter unavailable, using direct execution: %s", exc)

        # --- Direct execution (no router available) ---
        return {
            "current_subtask": next_subtask,
            "phase": AgentPhase.ACTING,
        }

    # No more sub-tasks — plan complete
    return {
        "current_subtask": None,
        "phase": AgentPhase.ACTING,
    }


# --- Internal: thread-safe ModelRouter singleton access ---

_model_router: Any = None


def _get_model_router() -> Any:
    """Return the module-level ModelRouter singleton (set at app startup)."""
    return _model_router


def set_model_router(router: Any) -> None:
    """Set the module-level ModelRouter singleton.

    Called from the FastAPI lifespan / application bootstrap so that
    every node invocation can access the router without threading issues.
    """
    global _model_router
    _model_router = router
    logger.info("ModelRouter installed in orchestration nodes")


# --- Internal: thread-safe SensorFusion singleton access ---

_sensor_fusion: Any = None


def _get_sensor_fusion() -> Any:
    """Return the module-level SensorFusion singleton (set at app startup)."""
    return _sensor_fusion


def set_sensor_fusion(sf: Any) -> None:
    """Set the module-level SensorFusion singleton.

    Called from ``create_agent_runner`` so that ``perceive_node`` can
    access real sensor data from MuJoCo (or future sensor backends).
    When not set, ``perceive_node`` falls back to stub perception data.
    """
    global _sensor_fusion
    _sensor_fusion = sf
    logger.info("SensorFusion installed in orchestration nodes")


# --- Internal: thread-safe TaskDecomposer LLM singleton access ---

_task_decomposer_llm: Any = None


def _get_task_decomposer_llm() -> Any:
    """Return the module-level LLM client for TaskDecomposer."""
    return _task_decomposer_llm


def set_task_decomposer_llm(llm: Any) -> None:
    """Set the module-level LLM client singleton for planning.

    Called from ``create_agent_runner`` or application bootstrap so
    that ``plan_node`` can pass a real LLM to ``TaskDecomposer``.
    When not set, ``TaskDecomposer`` falls back to template planning.
    """
    global _task_decomposer_llm
    _task_decomposer_llm = llm
    logger.info("TaskDecomposer LLM installed in orchestration nodes")


async def feedback_node(state: AgentGraphState) -> dict[str, Any]:
    """Evaluate sub-task outcome, detect failures, update episode.

    Routes to continue, recover, delegate, hri, or complete.
    """
    logger.debug(f"[{state.robot_id}] FEEDBACK phase")

    current = state.current_subtask
    action_result = state.last_action_result

    # If no current sub-task and no error, plan is complete
    if current is None and state.error is None:
        logger.info(f"[{state.robot_id}] All sub-tasks complete!")
        return {
            "phase": AgentPhase.EVALUATING,
            "should_continue": False,
        }

    if current:
        sub_task_id = current.get("sub_task_id", "")

        # Check actual execution outcome
        if action_result:
            status = action_result.get("status", "success")
            if status == "failure":
                error_detail = action_result.get("error", {})
                logger.warning(
                    f"[{state.robot_id}] Sub-task {sub_task_id} FAILED: {error_detail}"
                )
                return {
                    "phase": AgentPhase.EVALUATING,
                    "error": {
                        "type": "execution_failure",
                        "detail": f"Sub-task '{sub_task_id}' failed: {error_detail}",
                        "action_result": action_result,
                    },
                }

            # Success — mark as completed
            completed = list(state.completed_subtask_ids) + [sub_task_id]
            duration = action_result.get("duration_sec", 0)
            logger.info(
                f"[{state.robot_id}] Sub-task {sub_task_id} completed "
                f"({duration:.2f}s)"
            )
            return {
                "completed_subtask_ids": completed,
                "current_subtask": None,
                "last_action_result": None,
                "phase": AgentPhase.EVALUATING,
            }

        # No action result — simulated path (no executors wired)
        completed = list(state.completed_subtask_ids) + [sub_task_id]
        logger.info(
            f"[{state.robot_id}] Sub-task {sub_task_id} completed (simulated)"
        )
        return {
            "completed_subtask_ids": completed,
            "current_subtask": None,
            "last_action_result": None,
            "phase": AgentPhase.EVALUATING,
        }

    return {"phase": AgentPhase.EVALUATING}


async def recover_node(state: AgentGraphState) -> dict[str, Any]:
    """Attempt failure recovery using the chain-of-responsibility.

    In production, this calls roboclaw.recovery.RecoveryChain.
    """
    logger.warning(f"[{state.robot_id}] RECOVER phase — attempt {state.recovery_attempts + 1}")

    error = state.error or {}
    error_type = error.get("type", "unknown")

    # Simulated recovery logic
    recovery_attempts = state.recovery_attempts + 1

    if recovery_attempts >= state.max_recovery_attempts:
        logger.error(f"[{state.robot_id}] Max recovery attempts reached. Requesting human help.")
        return {
            "recovery_attempts": recovery_attempts,
            "phase": AgentPhase.RECOVERING,
            "require_human_help": True,
            "error": {**error, "recovery_failed": True, "attempts": recovery_attempts},
        }

    # Simulate successful recovery for known error types
    if error_type in ("kinematic", "dynamic", "perception"):
        logger.info(f"[{state.robot_id}] Recovery successful — will retry")
        return {
            "recovery_attempts": recovery_attempts,
            "phase": AgentPhase.RECOVERING,
            "error": None,  # Clear error for retry
        }

    # Unknown error — escalate
    return {
        "recovery_attempts": recovery_attempts,
        "phase": AgentPhase.RECOVERING,
        "require_human_help": True,
    }


async def delegate_a2a_node(state: AgentGraphState) -> dict[str, Any]:
    """Delegate a sub-task to a peer agent via A2A protocol."""
    logger.debug(f"[{state.robot_id}] DELEGATE phase")
    # In production: calls roboclaw.a2a.A2AClient
    return {"phase": AgentPhase.DELEGATING}


async def hri_node(state: AgentGraphState) -> dict[str, Any]:
    """Handle human-robot interaction (request help, respond to query)."""
    logger.debug(f"[{state.robot_id}] HRI phase")
    # In production: sends message via WebSocket, awaits operator response
    return {"phase": AgentPhase.HRI}
