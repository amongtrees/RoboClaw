"""Context compiler — assembles LLM prompt context from all memory tiers.

This is the bridge that makes planning intelligent: it queries working memory
for current state, episodic memory for past experiences, semantic memory for
similar tasks, spatial memory for environment layout, and the knowledge base
for domain facts — then packages everything into a structured LLM prompt.
"""

from __future__ import annotations

from typing import Any

from roboclaw.models.memory import LLMContext
from roboclaw.models.state import HumanoidState


class ContextCompiler:
    """Assembles structured context for LLM agent calls from all memory tiers."""

    def __init__(
        self,
        working_memory: Any = None,
        episodic_memory: Any = None,
        semantic_memory: Any = None,
        spatial_memory: Any = None,
        knowledge_base: Any = None,
    ) -> None:
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = semantic_memory
        self._sp = spatial_memory
        self._kb = knowledge_base

    async def compile(
        self,
        robot_id: str,
        task_goal: str,
        current_state: HumanoidState | None = None,
        query_embedding: list[float] | None = None,
    ) -> LLMContext:
        """Assemble full LLM context from all memory sources.

        Args:
            robot_id: The robot's identifier.
            task_goal: The high-level task description.
            current_state: Current robot state (from working memory if None).
            query_embedding: Embedding of the task goal for semantic search.
        """
        # Current state
        if current_state is None and self._wm:
            current_state = await self._wm.get_robot_state()

        state_summary = self._summarize_state(current_state) if current_state else "Unknown"

        # Episodic: similar past episodes
        relevant_episodes = []
        if self._em:
            try:
                relevant_episodes = await self._em.search_episodes(
                    robot_id=robot_id,
                    limit=5,
                )
            except Exception:
                pass

        # Semantic: similar tasks, skills, failure patterns
        relevant_semantic: list[dict[str, Any]] = []
        if self._sm and query_embedding:
            try:
                relevant_semantic = await self._sm.find_similar_episodes(query_embedding, top_k=5)
            except Exception:
                pass

        # Knowledge base: affordances and safety rules
        relevant_knowledge: list[dict[str, Any]] = []
        if self._kb:
            try:
                safety_rules = await self._kb.get_safety_rules()
                relevant_knowledge.extend(safety_rules)
                # Extract object labels from the task goal (simple heuristic)
                for word in task_goal.lower().split():
                    affordance = await self._kb.get_affordance(word)
                    if affordance:
                        relevant_knowledge.append({"key": f"affordance:{word}", "value": str(affordance)})
            except Exception:
                pass

        # Safety constraints
        safety_constraints = [
            "ZMP must remain within support polygon at all times",
            "Joint torques must not exceed configured limits",
            "Collision force must stay below threshold",
            "Maintain stable foot contact during manipulation",
        ]
        if self._kb:
            try:
                rules = await self._kb.get_safety_rules()
                safety_constraints.extend(
                    r.get("description", str(r)) for r in rules[:5]
                )
            except Exception:
                pass

        # Spatial context
        spatial_context: dict[str, Any] = {}
        if self._sp:
            try:
                spatial_context = {
                    "current_room": self._sp.current_room,
                    "robot_position": self._sp.robot_position,
                    "rooms": list(self._sp._rooms.keys()) if hasattr(self._sp, "_rooms") else [],
                }
            except Exception:
                pass

        return LLMContext(
            robot_id=robot_id,
            task_goal=task_goal,
            current_state_summary=state_summary,
            current_phase=current_state.phase if current_state else "idle",
            relevant_episodes=relevant_episodes,
            relevant_semantic=relevant_semantic,
            relevant_knowledge=relevant_knowledge,
            spatial_context=spatial_context,
            safety_constraints=safety_constraints,
        )

    @staticmethod
    def _summarize_state(state: HumanoidState) -> str:
        """Create a concise text summary of the current robot state."""
        parts = [
            f"Robot: {state.robot_id}",
            f"Phase: {state.phase or 'idle'}",
            f"Joint count: {len(state.joint_states)}",
        ]

        if state.zmp != (0.0, 0.0):
            parts.append(f"ZMP: ({state.zmp[0]:.3f}, {state.zmp[1]:.3f})")

        if state.end_effector_pose_left:
            p = state.end_effector_pose_left.position
            parts.append(f"Left EE: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
        if state.end_effector_pose_right:
            p = state.end_effector_pose_right.position
            parts.append(f"Right EE: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

        parts.append(f"Support: {state.support_phase}")
        parts.append(f"Objects visible: {len(state.visible_objects)}")

        if state.visible_objects:
            obj_labels = [o.get("label", "?") for o in state.visible_objects[:10]]
            parts.append(f"Detected: {', '.join(obj_labels)}")

        return "\n".join(parts)
