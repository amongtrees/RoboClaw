"""LLM-driven task decomposition — breaks high-level goals into executable SubTask DAGs."""

from __future__ import annotations

import json
import logging
from typing import Any

from roboclaw.core.types import SkillType, model_type_for_skill
from roboclaw.models.memory import LLMContext
from roboclaw.models.plan import ExecutionPlan, SubTask
from roboclaw.planning.base import AbstractPlanner
from roboclaw.planning.plan_validator import PlanValidator

logger = logging.getLogger(__name__)


class TaskDecomposer(AbstractPlanner):
    """Decomposes natural language tasks into executable SubTask DAGs using an LLM.

    Uses LangChain to call the LLM with compiled context from ContextCompiler.
    The LLM is prompted to output a structured JSON plan with preconditions.
    """

    SYSTEM_PROMPT = """You are a humanoid robot task planner. Your job is to decompose high-level natural language
tasks into sequences of executable sub-tasks for a bipedal humanoid robot with dual arms.

The robot has these skills available:
- navigate_to(room: str) — Walk to a named room
- walk_to(pose: dict) — Walk to a specific pose {x, y, z, qx, qy, qz, qw}
- locate_object(object: str) — Find and localize an object in the scene
- whole_body_grasp(object: str, arm: str, grasp_type: str) — Grasp an object with whole-body coordination
- place_object(object: str, target: dict) — Place an object at a location
- handover(object: str, receiver: str) — Hand an object to a person or agent
- speak(text: str) — Speak text to a human
- gaze_at(target: str) — Look at a target
- open_door(door: str) — Open a door
- press_button(button: str) — Press a button
- wait_for(condition: str) — Wait for a condition

Output a JSON object with this structure:
{
  "plan_id": "string",
  "task_goal": "string",
  "estimated_duration_sec": float,
  "sub_tasks": [
    {
      "sub_task_id": "string (unique, descriptive)",
      "skill_type": "string (one of the skills above)",
      "parameters": {"key": "value"},
      "preconditions": ["sub_task_ids that must complete first"],
      "expected_outcome": {"key": "value"},
      "timeout_sec": float,
      "priority": int,
      "retry_policy": "default"
    }
  ]
}

Rules:
1. Each sub_task_id must be unique and descriptive (e.g., "navigate_to_kitchen", "grasp_cup_right_arm")
2. Preconditions form a valid DAG — no cycles
3. Always start with perception (e.g., gaze_at or locate_object) to understand the scene
4. Include safety checks between major actions
5. Consider the robot's bipedal balance — avoid simultaneous walking and manipulation
6. Be specific about which arm to use for manipulation tasks
7. Estimate realistic timeouts based on the action complexity
"""

    def __init__(self, llm: Any = None, validator: PlanValidator | None = None) -> None:
        self._llm = llm
        self._validator = validator or PlanValidator()

    async def plan(
        self,
        task_spec: dict[str, Any],
        world_state: dict[str, Any],
        memory_context: Any,
    ) -> tuple[ExecutionPlan, dict[str, str]]:
        """Decompose a task using LLM or fallback to template-based planning.

        Returns:
            (execution_plan, routing_map) where routing_map is
            ``{sub_task_id: ModelType_value}`` for every sub-task.
        """
        goal = task_spec.get("goal", "")

        if self._llm:
            plan, routing = await self._llm_plan(goal, memory_context)
        else:
            plan, routing = self._template_plan(goal, task_spec)

        # Annotate every SubTask's parameters with its _model_type
        for st in plan.sub_tasks:
            if st.sub_task_id in routing:
                st.parameters["_model_type"] = routing[st.sub_task_id]
            elif "_model_type" not in st.parameters:
                model_type = model_type_for_skill(st.skill_type)
                st.parameters["_model_type"] = model_type
                routing[st.sub_task_id] = model_type

        return plan, routing

    async def _llm_plan(self, goal: str, context: Any) -> tuple[ExecutionPlan, dict[str, str]]:
        """Use LLM to generate a plan.  Returns (plan, routing_map)."""
        context_text = ""
        if isinstance(context, LLMContext):
            context_text = context.to_prompt_text()
        elif context:
            context_text = str(context)

        user_prompt = f"Task Goal: {goal}\n\nContext:\n{context_text}"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = await self._llm.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                plan_dict = json.loads(content[json_start:json_end])
                plan = ExecutionPlan(**plan_dict)
                # Validate
                plan = await self._validator.validate(plan)
                routing = self._build_routing_map(plan)
                return plan, routing
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, falling back to template planning")

        return self._template_plan(goal, {})

    def _template_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Generate a plan using templates for common task patterns."""
        goal_lower = goal.lower()

        # Pattern: "go to X and pick up Y"
        if "pick up" in goal_lower or "grasp" in goal_lower:
            return self._pick_and_place_plan(goal, task_spec)

        # Pattern: "go to X"
        if "go to" in goal_lower or "navigate to" in goal_lower:
            return self._navigation_plan(goal, task_spec)

        # Pattern: "find X" or "look for X"
        if "find" in goal_lower or "look for" in goal_lower or "search" in goal_lower:
            return self._search_plan(goal, task_spec)

        # Generic fallback plan
        return self._generic_plan(goal, task_spec)

    def _pick_and_place_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for pick-and-place tasks."""
        import re

        # Extract target object and destination from goal
        object_match = re.search(r"(?:pick up|grasp|grab)\s+(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        room_match = re.search(r"(?:go to|in|from)\s+(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        target_object = object_match.group(1) if object_match else "object"
        target_room = room_match.group(1) if room_match else "kitchen"

        perceive_id = "perceive_scene"
        nav_id = f"navigate_to_{target_room}"
        locate_id = f"locate_{target_object}"
        grasp_id = f"grasp_{target_object}"

        sub_tasks = [
            SubTask(
                sub_task_id=perceive_id,
                skill_type=SkillType.GAZE_AT,
                parameters={"target": "scene", "_model_type": "llm"},
                expected_outcome={"objects_detected": True},
                timeout_sec=5.0,
                priority=0,
            ),
            SubTask(
                sub_task_id=nav_id,
                skill_type=SkillType.NAVIGATE_TO,
                parameters={"room": target_room, "_model_type": "vln"},
                preconditions=[perceive_id],
                expected_outcome={"at_location": target_room},
                timeout_sec=60.0,
                priority=1,
            ),
            SubTask(
                sub_task_id=locate_id,
                skill_type=SkillType.GAZE_AT,
                parameters={"target": target_object, "_model_type": "llm"},
                preconditions=[nav_id],
                expected_outcome={"object_found": True},
                timeout_sec=10.0,
                priority=2,
            ),
            SubTask(
                sub_task_id=grasp_id,
                skill_type=SkillType.WHOLE_BODY_GRASP,
                parameters={"object": target_object, "arm": "right", "grasp_type": "top_down_encompassing", "_model_type": "vla"},
                preconditions=[locate_id],
                expected_outcome={"grasped": True},
                timeout_sec=15.0,
                priority=3,
            ),
        ]

        routing = {
            perceive_id: "llm",
            nav_id: "vln",
            locate_id: "llm",
            grasp_id: "vla",
        }

        return ExecutionPlan(
            plan_id=f"plan_pick_{target_object}",
            task_goal=goal,
            sub_tasks=sub_tasks,
            estimated_duration_sec=90.0,
        ), routing

    def _navigation_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for navigation-only tasks."""
        import re

        room_match = re.search(r"(?:go to|navigate to)\s+(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        target_room = room_match.group(1) if room_match else "living_room"

        perceive_id = "perceive_scene"
        nav_id = f"navigate_to_{target_room}"

        sub_tasks = [
            SubTask(
                sub_task_id=perceive_id,
                skill_type=SkillType.GAZE_AT,
                parameters={"target": "scene", "_model_type": "llm"},
                expected_outcome={"objects_detected": True},
                timeout_sec=5.0,
                priority=0,
            ),
            SubTask(
                sub_task_id=nav_id,
                skill_type=SkillType.NAVIGATE_TO,
                parameters={"room": target_room, "_model_type": "vln"},
                preconditions=[perceive_id],
                expected_outcome={"at_location": target_room},
                timeout_sec=60.0,
                priority=1,
            ),
        ]

        routing = {perceive_id: "llm", nav_id: "vln"}

        return ExecutionPlan(
            plan_id=f"plan_nav_{target_room}",
            task_goal=goal,
            sub_tasks=sub_tasks,
            estimated_duration_sec=65.0,
        ), routing

    def _search_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for search tasks."""
        import re

        obj_match = re.search(r"(?:find|look for|search for)\s+(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        target_object = obj_match.group(1) if obj_match else "target"

        survey_id = "survey_room"
        search_id = f"search_for_{target_object}"

        sub_tasks = [
            SubTask(
                sub_task_id=survey_id,
                skill_type=SkillType.GAZE_AT,
                parameters={"target": "panorama", "_model_type": "llm"},
                expected_outcome={"room_surveyed": True},
                timeout_sec=15.0,
                priority=0,
            ),
            SubTask(
                sub_task_id=search_id,
                skill_type=SkillType.NAVIGATE_TO,
                parameters={"target": "search_pattern", "_model_type": "vln"},
                preconditions=[survey_id],
                expected_outcome={"object_found": True},
                timeout_sec=30.0,
                priority=1,
            ),
        ]

        routing = {survey_id: "llm", search_id: "vln"}

        return ExecutionPlan(
            plan_id=f"plan_search_{target_object}",
            task_goal=goal,
            sub_tasks=sub_tasks,
            estimated_duration_sec=45.0,
        ), routing

    def _generic_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Generic fallback plan for unknown task patterns."""
        perceive_id = "perceive_scene"
        speak_id = "speak_confirmation"

        sub_tasks = [
            SubTask(
                sub_task_id=perceive_id,
                skill_type=SkillType.GAZE_AT,
                parameters={"target": "scene", "_model_type": "llm"},
                expected_outcome={"objects_detected": True},
                timeout_sec=5.0,
                priority=0,
            ),
            SubTask(
                sub_task_id=speak_id,
                skill_type=SkillType.SPEAK,
                parameters={"text": f"I'll help you with: {goal}", "_model_type": "llm"},
                preconditions=[perceive_id],
                expected_outcome={"spoken": True},
                timeout_sec=3.0,
                priority=1,
            ),
        ]

        routing = {perceive_id: "llm", speak_id: "llm"}

        return ExecutionPlan(
            plan_id=f"plan_generic",
            task_goal=goal,
            sub_tasks=sub_tasks,
            estimated_duration_sec=10.0,
        ), routing

    # ------------------------------------------------------------------
    # Routing map builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_routing_map(plan: ExecutionPlan) -> dict[str, str]:
        """Build a ``{sub_task_id: ModelType}`` dict from a plan's sub-tasks.

        Reads the explicit ``_model_type`` annotation in each SubTask's
        parameters, falling back to the static skill→model lookup table.
        """
        routing: dict[str, str] = {}
        for st in plan.sub_tasks:
            if st.sub_task_id in routing:
                continue
            explicit = st.parameters.get("_model_type")
            if explicit:
                routing[st.sub_task_id] = explicit
            else:
                routing[st.sub_task_id] = model_type_for_skill(st.skill_type)
        return routing
