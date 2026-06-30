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
tasks into sequences of executable sub-tasks for a bipedal humanoid robot (Unitree H1) with dual arms.

The robot has these 9 skills available, all executed with real MuJoCo physics:
- navigate_to(room: str) — Walk to a named room (kitchen, living_room, bedroom, etc.)
- walk_steps(num_steps: int, direction: str, step_length_m: float) — Walk N steps forward/backward/left/right
- whole_body_grasp(object: str, arm: str, grasp_type: str) — Grasp an object with whole-body coordination. arm: left|right, grasp_type: top_down_encompassing
- place_object(object: str, target_x: float, target_y: float, target_z: float) — Place an object at a location
- handover(object: str, receiver: str) — Hand an object to a person or agent
- speak(text: str) — Speak text via TTS
- gaze_at(target: str) — Look at a target (uses torso rotation)
- open_door(door: str, arm: str) — Open a door with whole-body coordination
- climb_stairs(num_steps: int, direction: str) — Ascend or descend stairs

IMPORTANT CONSTRAINTS:
1. The robot is bipedal — it CANNOT walk and manipulate simultaneously. Separate them into sequential steps.
2. Always start with gaze_at to perceive the scene before other actions.
3. Leg skills (navigate_to, walk_steps, climb_stairs) require both legs — only one leg/locomotion skill at a time.
4. Arm skills (whole_body_grasp, place_object, handover, open_door) use arms — the robot must be stationary (not walking) during arm actions.
5. Each sub_task_id must be unique and descriptive (e.g., "navigate_to_kitchen", "grasp_cup_right_arm")
6. Preconditions form a valid DAG — no cycles.
7. Be specific about which arm to use for manipulation tasks.
8. The robot speaks to confirm task completion.

Output a JSON object with this structure:
{
  "plan_id": "string",
  "task_goal": "string",
  "estimated_duration_sec": float,
  "sub_tasks": [
    {
      "sub_task_id": "string (unique, descriptive)",
      "skill_type": "string (one of the 9 skills above)",
      "parameters": {"key": "value"},
      "preconditions": ["sub_task_ids that must complete first"],
      "expected_outcome": {"key": "value"},
      "timeout_sec": float,
      "priority": int,
      "retry_policy": "default"
    }
  ]
}
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

        # Pattern: "go to X", "walk to X"
        if ("go to" in goal_lower or "navigate to" in goal_lower
                or "walk to" in goal_lower):
            return self._navigation_plan(goal, task_spec)

        # Pattern: "walk N steps", "step forward/backward", "move left/right"
        if any(w in goal_lower for w in ("walk", "step forward", "step backward",
                                          "move forward", "move backward",
                                          "move left", "move right",
                                          "step left", "step right")):
            return self._locomotion_plan(goal, task_spec)

        # Pattern: "climb stairs"
        if "climb" in goal_lower or "stairs" in goal_lower:
            return self._climb_plan(goal, task_spec)

        # Pattern: "open door"
        if "open" in goal_lower and "door" in goal_lower:
            return self._door_plan(goal, task_spec)

        # Pattern: "hand over" / "handover"
        if "hand" in goal_lower and ("over" in goal_lower or "me" in goal_lower):
            return self._handover_plan(goal, task_spec)

        # Pattern: "place" / "put down"
        if "place" in goal_lower or "put" in goal_lower:
            return self._place_plan(goal, task_spec)

        # Pattern: "speak" / "say"
        if "speak" in goal_lower or "say" in goal_lower:
            return self._speak_plan(goal, task_spec)

        # Pattern: "look at" / "gaze"
        if "look at" in goal_lower or "gaze" in goal_lower:
            return self._gaze_plan(goal, task_spec)

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

    def _locomotion_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for walking / stepping commands."""
        import re

        # Extract number of steps
        num_match = re.search(r"(\d+)\s*steps?", goal, re.IGNORECASE)
        num_steps = int(num_match.group(1)) if num_match else 3

        # Extract direction
        direction = "forward"
        if "backward" in goal.lower():
            direction = "backward"
        elif "left" in goal.lower():
            direction = "left"
        elif "right" in goal.lower():
            direction = "right"

        # Extract step length
        length_match = re.search(r"(\d+\.?\d*)\s*m(?:eter)?", goal, re.IGNORECASE)
        step_length = float(length_match.group(1)) if length_match else 0.3

        walk_id = f"walk_{direction}_{num_steps}steps"

        sub_tasks = [
            SubTask(
                sub_task_id=walk_id,
                skill_type=SkillType.WALK_STEPS,
                parameters={
                    "num_steps": num_steps,
                    "direction": direction,
                    "step_length_m": step_length,
                    "_model_type": "vln",
                },
                expected_outcome={"steps_completed": num_steps},
                timeout_sec=num_steps * 10.0,
                priority=0,
            ),
        ]

        routing = {walk_id: "vln"}

        return ExecutionPlan(
            plan_id=f"plan_walk_{direction}",
            task_goal=goal,
            sub_tasks=sub_tasks,
            estimated_duration_sec=num_steps * 10.0,
        ), routing

    def _climb_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for stair climbing."""
        import re
        num_match = re.search(r"(\d+)\s*st(?:air|ep)s?", goal, re.IGNORECASE)
        num_stairs = int(num_match.group(1)) if num_match else 5
        direction = "down" if "down" in goal.lower() else "up"

        climb_id = "climb_stairs_task"
        sub_tasks = [
            SubTask(
                sub_task_id=climb_id,
                skill_type=SkillType.CLIMB_STAIRS,
                parameters={"num_steps": num_stairs, "direction": direction, "_model_type": "vln"},
                expected_outcome={"stairs_climbed": True},
                timeout_sec=num_stairs * 10.0,
                priority=0,
            ),
        ]
        routing = {climb_id: "vln"}
        return ExecutionPlan(
            plan_id="plan_climb", task_goal=goal,
            sub_tasks=sub_tasks, estimated_duration_sec=num_stairs * 10.0,
        ), routing

    def _door_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for opening a door."""
        import re
        door_match = re.search(r"(?:open\s+)?(?:the\s+)?(\w+)\s*door", goal, re.IGNORECASE)
        door_name = door_match.group(1) if door_match else "main_door"
        arm = "left" if "left" in goal.lower() else "right"

        nav_id = "approach_door"
        open_id = "open_door_task"
        sub_tasks = [
            SubTask(sub_task_id=nav_id, skill_type=SkillType.NAVIGATE_TO,
                    parameters={"room": "doorway", "_model_type": "vln"},
                    expected_outcome={"at_location": "doorway"}, timeout_sec=30.0, priority=0),
            SubTask(sub_task_id=open_id, skill_type=SkillType.OPEN_DOOR,
                    parameters={"door": door_name, "arm": arm, "_model_type": "vla"},
                    preconditions=[nav_id],
                    expected_outcome={"door_opened": True}, timeout_sec=45.0, priority=1),
        ]
        routing = {nav_id: "vln", open_id: "vla"}
        return ExecutionPlan(
            plan_id=f"plan_open_{door_name}", task_goal=goal,
            sub_tasks=sub_tasks, estimated_duration_sec=75.0,
        ), routing

    def _handover_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for handover tasks."""
        import re
        obj_match = re.search(r"(?:hand|give|pass)\s+(?:over\s+)?(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        obj = obj_match.group(1) if obj_match else "object"
        arm = "left" if "left" in goal.lower() else "right"

        grasp_id = f"grasp_{obj}"
        handover_id = f"handover_{obj}"
        sub_tasks = [
            SubTask(sub_task_id=grasp_id, skill_type=SkillType.WHOLE_BODY_GRASP,
                    parameters={"object": obj, "arm": arm, "_model_type": "vla"},
                    expected_outcome={"grasped": True}, timeout_sec=15.0, priority=0),
            SubTask(sub_task_id=handover_id, skill_type=SkillType.HANDOVER,
                    parameters={"object": obj, "receiver": "human", "_model_type": "vla"},
                    preconditions=[grasp_id],
                    expected_outcome={"handed_over": True}, timeout_sec=20.0, priority=1),
        ]
        routing = {grasp_id: "vla", handover_id: "vla"}
        return ExecutionPlan(
            plan_id=f"plan_handover_{obj}", task_goal=goal,
            sub_tasks=sub_tasks, estimated_duration_sec=35.0,
        ), routing

    def _place_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for placing an object."""
        import re
        obj_match = re.search(r"(?:place|put)\s+(?:down\s+)?(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        obj = obj_match.group(1) if obj_match else "object"
        x_match = re.search(r"x[=:\s]*(-?\d+\.?\d*)", goal, re.IGNORECASE)
        y_match = re.search(r"y[=:\s]*(-?\d+\.?\d*)", goal, re.IGNORECASE)
        z_match = re.search(r"z[=:\s]*(-?\d+\.?\d*)", goal, re.IGNORECASE)
        tx = float(x_match.group(1)) if x_match else 0.3
        ty = float(y_match.group(1)) if y_match else 0.0
        tz = float(z_match.group(1)) if z_match else 0.5

        place_id = f"place_{obj}"
        sub_tasks = [
            SubTask(sub_task_id=place_id, skill_type=SkillType.PLACE_OBJECT,
                    parameters={"object": obj, "target_x": tx, "target_y": ty, "target_z": tz, "_model_type": "vla"},
                    expected_outcome={"placed": True}, timeout_sec=20.0, priority=0),
        ]
        routing = {place_id: "vla"}
        return ExecutionPlan(
            plan_id=f"plan_place_{obj}", task_goal=goal,
            sub_tasks=sub_tasks, estimated_duration_sec=20.0,
        ), routing

    def _speak_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for speaking."""
        import re
        text_match = re.search(r"(?:speak|say)\s+(.+)", goal, re.IGNORECASE)
        text = text_match.group(1).strip() if text_match else goal

        speak_id = "speak_task"
        sub_tasks = [
            SubTask(sub_task_id=speak_id, skill_type=SkillType.SPEAK,
                    parameters={"text": text, "_model_type": "llm"},
                    expected_outcome={"spoken": True}, timeout_sec=5.0, priority=0),
        ]
        routing = {speak_id: "llm"}
        return ExecutionPlan(
            plan_id="plan_speak", task_goal=goal,
            sub_tasks=sub_tasks, estimated_duration_sec=5.0,
        ), routing

    def _gaze_plan(self, goal: str, task_spec: dict[str, Any]) -> tuple[ExecutionPlan, dict[str, str]]:
        """Template plan for gaze commands."""
        import re
        target_match = re.search(r"(?:look at|gaze at|gaze)\s+(?:the\s+)?(\w+)", goal, re.IGNORECASE)
        target = target_match.group(1) if target_match else "scene"

        gaze_id = f"gaze_{target}"
        sub_tasks = [
            SubTask(sub_task_id=gaze_id, skill_type=SkillType.GAZE_AT,
                    parameters={"target": target, "_model_type": "llm"},
                    expected_outcome={"gazed": True}, timeout_sec=5.0, priority=0),
        ]
        routing = {gaze_id: "llm"}
        return ExecutionPlan(
            plan_id=f"plan_gaze_{target}", task_goal=goal,
            sub_tasks=sub_tasks, estimated_duration_sec=5.0,
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
