"""Skill library — registry of named, parameterized robot skills.

Each skill wraps an ActionExecutor call with pre/post conditions.
Skills can be bound as LangChain tools for LLM-driven invocation.
"""

from __future__ import annotations

import logging
from time import time
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus
from roboclaw.core.types import SkillType
from roboclaw.models.skill import SkillDefinition, SkillParameter

logger = logging.getLogger(__name__)


class SkillLibrary:
    """Registry of parameterized robot skills.

    Maps skill types to their definitions and executor implementations.
    Supports LangChain tool binding for LLM-driven skill selection.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._executors: dict[str, AbstractActionExecutor] = {}
        self._register_builtin_skills()

    def _register_builtin_skills(self) -> None:
        """Register the standard humanoid robot skill catalog."""
        builtins = [
            SkillDefinition(
                name="Navigate to Room",
                skill_type=SkillType.NAVIGATE_TO,
                description="Walk to a named room in the environment.",
                parameters=[
                    SkillParameter(name="room", param_type="string", description="Target room name", required=True),
                    SkillParameter(name="speed", param_type="float", description="Walking speed in m/s", required=False, default=1.0),
                ],
                timeout_sec=120.0,
                tags=["navigation", "locomotion"],
            ),
            SkillDefinition(
                name="Walk Steps",
                skill_type=SkillType.WALK_STEPS,
                description="Walk a specified number of steps in a direction.",
                parameters=[
                    SkillParameter(name="num_steps", param_type="int", description="Number of steps", required=True),
                    SkillParameter(name="direction", param_type="string", description="forward | backward | left | right", required=False, default="forward"),
                    SkillParameter(name="step_length_m", param_type="float", description="Step length in meters", required=False, default=0.3),
                ],
                timeout_sec=30.0,
                tags=["locomotion"],
            ),
            SkillDefinition(
                name="Whole-Body Grasp",
                skill_type=SkillType.WHOLE_BODY_GRASP,
                description="Grasp an object using whole-body coordination (balance + arm + hand).",
                parameters=[
                    SkillParameter(name="object", param_type="string", description="Object label to grasp", required=True),
                    SkillParameter(name="arm", param_type="string", description="Which arm: left | right", required=False, default="right"),
                    SkillParameter(name="grasp_type", param_type="string", description="Grasp strategy", required=False, default="top_down_encompassing"),
                ],
                timeout_sec=30.0,
                tags=["manipulation", "grasping"],
            ),
            SkillDefinition(
                name="Place Object",
                skill_type=SkillType.PLACE_OBJECT,
                description="Place a held object at a target location.",
                parameters=[
                    SkillParameter(name="object", param_type="string", description="Object label", required=True),
                    SkillParameter(name="target_x", param_type="float", description="Target X position", required=True),
                    SkillParameter(name="target_y", param_type="float", description="Target Y position", required=True),
                    SkillParameter(name="target_z", param_type="float", description="Target Z position", required=True),
                ],
                timeout_sec=20.0,
                tags=["manipulation"],
            ),
            SkillDefinition(
                name="Handover",
                skill_type=SkillType.HANDOVER,
                description="Hand an object to a person or another agent.",
                parameters=[
                    SkillParameter(name="object", param_type="string", description="Object to hand over", required=True),
                    SkillParameter(name="receiver", param_type="string", description="Receiver: human | agent_id", required=True),
                ],
                timeout_sec=30.0,
                tags=["manipulation", "hri"],
            ),
            SkillDefinition(
                name="Speak",
                skill_type=SkillType.SPEAK,
                description="Speak text to a human via TTS.",
                parameters=[
                    SkillParameter(name="text", param_type="string", description="Text to speak", required=True),
                    SkillParameter(name="volume", param_type="float", description="Volume 0.0-1.0", required=False, default=0.8),
                ],
                timeout_sec=10.0,
                tags=["hri", "communication"],
            ),
            SkillDefinition(
                name="Gaze At",
                skill_type=SkillType.GAZE_AT,
                description="Orient head/camera to look at a target.",
                parameters=[
                    SkillParameter(name="target", param_type="string", description="Target to look at", required=True),
                ],
                timeout_sec=5.0,
                tags=["perception"],
            ),
            SkillDefinition(
                name="Open Door",
                skill_type=SkillType.OPEN_DOOR,
                description="Open a door using whole-body coordination.",
                parameters=[
                    SkillParameter(name="door", param_type="string", description="Door identifier", required=True),
                    SkillParameter(name="arm", param_type="string", description="Arm to use", required=False, default="right"),
                ],
                timeout_sec=45.0,
                tags=["manipulation", "navigation"],
            ),
            SkillDefinition(
                name="Climb Stairs",
                skill_type=SkillType.CLIMB_STAIRS,
                description="Ascend or descend stairs using bipedal locomotion.",
                parameters=[
                    SkillParameter(name="num_steps", param_type="int", description="Number of stairs", required=True),
                    SkillParameter(name="direction", param_type="string", description="up | down", required=False, default="up"),
                ],
                timeout_sec=60.0,
                tags=["locomotion"],
            ),
        ]

        for skill in builtins:
            self._skills[skill.skill_type] = skill

    def register_executor(self, skill_type: str, executor: AbstractActionExecutor) -> None:
        """Register an executor implementation for a skill type."""
        self._executors[skill_type] = executor

    def get_definition(self, skill_type: str) -> SkillDefinition | None:
        """Get the definition for a skill type."""
        return self._skills.get(skill_type)

    def list_skills(self) -> list[SkillDefinition]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_skill_names(self) -> list[str]:
        """List all skill type names."""
        return list(self._skills.keys())

    async def execute_skill(
        self,
        skill_type: str,
        parameters: dict[str, Any],
        world_state: dict[str, Any] | None = None,
        model_result: Any = None,
    ) -> ActionResult:
        """Execute a skill by its type name.

        Dispatches to the registered executor for that skill type.
        When ``model_result`` (a ``ModelInferenceResponse``) is provided
        and successful, its outputs drive the action result directly.
        """
        definition = self._skills.get(skill_type)
        if definition is None:
            return ActionResult(
                sub_task_id=skill_type,
                status=ActionResultStatus.FAILURE,
                error={"type": "unknown_skill", "detail": f"Skill '{skill_type}' not found"},
            )

        # --- Model-driven execution ---
        if model_result is not None and hasattr(model_result, "is_success") and model_result.is_success:
            logger.info(
                "Model-driven execution of skill '%s': confidence=%.2f, time=%.0fms",
                skill_type, model_result.confidence, model_result.inference_time_ms,
            )
            return ActionResult(
                sub_task_id=skill_type,
                status=ActionResultStatus.SUCCESS,
                duration_sec=model_result.inference_time_ms / 1000.0,
                actual_outcome={
                    "model_driven": True,
                    "skill": skill_type,
                    "params": parameters,
                    "waypoints": model_result.waypoints,
                    "joint_actions": model_result.joint_actions,
                    "ee_poses": model_result.ee_poses,
                    "gripper_command": model_result.gripper_command,
                    "confidence": model_result.confidence,
                },
            )

        executor = self._executors.get(skill_type)
        if executor is None:
            # No dedicated executor — use a simulated result for Phase 1
            logger.info(f"Simulating execution of skill: {skill_type} with params: {parameters}")
            return ActionResult(
                sub_task_id=skill_type,
                status=ActionResultStatus.SUCCESS,
                duration_sec=1.0,
                actual_outcome={"simulated": True, "skill": skill_type, "params": parameters},
            )

        start_time = time()
        try:
            result = await executor.execute(
                {"skill_type": skill_type, "parameters": parameters},
                world_state or {},
            )
            return result
        except Exception as e:
            duration = time() - start_time
            logger.error(f"Skill '{skill_type}' execution failed: {e}")
            return ActionResult(
                sub_task_id=skill_type,
                status=ActionResultStatus.FAILURE,
                duration_sec=duration,
                error={"type": "execution_error", "detail": str(e)},
            )

    # --- LangChain tool binding ---

    def as_langchain_tools(self) -> list:
        """Export skills as LangChain StructuredTool objects for LLM agent use."""
        try:
            from langchain_core.tools import StructuredTool

            tools = []
            for skill_type, definition in self._skills.items():
                async def _execute(**kwargs: Any) -> str:
                    st = skill_type  # capture
                    result = await self.execute_skill(st, kwargs)
                    return f"Skill '{st}': {result.status} — {result.actual_outcome}"

                tool = StructuredTool(
                    name=skill_type,
                    description=definition.description,
                    coroutine=_execute,
                )
                tools.append(tool)
            return tools
        except ImportError:
            logger.warning("LangChain not available; cannot create tools")
            return []
