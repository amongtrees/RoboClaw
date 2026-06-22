"""Plan validator — checks generated plans for feasibility and safety."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.core.errors import PlanValidationError
from roboclaw.models.plan import ExecutionPlan

logger = logging.getLogger(__name__)


class PlanValidator:
    """Validates execution plans against physical and safety constraints."""

    def __init__(self, robot_config: Any = None) -> None:
        self._robot_config = robot_config

    async def validate(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Validate a plan and raise PlanValidationError if invalid.

        Returns the plan unchanged if valid (allows chaining).
        """
        violations = []

        # Check for duplicate sub-task IDs
        ids = [st.sub_task_id for st in plan.sub_tasks]
        if len(ids) != len(set(ids)):
            violations.append("Duplicate sub-task IDs found")

        # Check for valid DAG (no cycles, valid preconditions)
        if not self._is_valid_dag(plan.sub_tasks):
            violations.append("Sub-task preconditions contain cycles")

        # Check that all preconditions reference existing sub-tasks
        valid_ids = set(ids)
        for st in plan.sub_tasks:
            for precond in st.preconditions:
                if precond not in valid_ids:
                    violations.append(f"Sub-task '{st.sub_task_id}' references unknown precondition '{precond}'")

        # Check for reasonable timeouts
        for st in plan.sub_tasks:
            if st.timeout_sec <= 0:
                violations.append(f"Sub-task '{st.sub_task_id}' has invalid timeout: {st.timeout_sec}s")
            if st.timeout_sec > 600:
                violations.append(f"Sub-task '{st.sub_task_id}' has excessive timeout: {st.timeout_sec}s")

        # Check for bipedal balance constraint: no simultaneous walk + manipulate
        for i, st_a in enumerate(plan.sub_tasks):
            for st_b in plan.sub_tasks[i + 1 :]:
                if self._are_parallel(st_a, st_b, plan.sub_tasks):
                    if self._is_walking(st_a) and self._is_manipulating(st_b):
                        violations.append(
                            f"Cannot perform '{st_a.sub_task_id}' (walking) and "
                            f"'{st_b.sub_task_id}' (manipulation) simultaneously"
                        )

        if violations:
            raise PlanValidationError(plan.plan_id, violations)

        return plan

    @staticmethod
    def _is_valid_dag(sub_tasks: list) -> bool:
        """Check for cycles in sub-task DAG using DFS."""
        ids = {st.sub_task_id: st for st in sub_tasks}
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            task = ids.get(node_id)
            if task:
                for precond in task.preconditions:
                    if precond not in visited:
                        if _has_cycle(precond):
                            return True
                    elif precond in rec_stack:
                        return True
            rec_stack.discard(node_id)
            return False

        for task_id in ids:
            if task_id not in visited:
                if _has_cycle(task_id):
                    return False
        return True

    @staticmethod
    def _are_parallel(task_a: Any, task_b: Any, all_tasks: list) -> bool:
        """Check if two tasks could execute in parallel (neither depends on the other)."""
        a_deps = set(task_a.preconditions)
        b_deps = set(task_b.preconditions)
        return task_a.sub_task_id not in b_deps and task_b.sub_task_id not in a_deps

    @staticmethod
    def _is_walking(task: Any) -> bool:
        return task.skill_type in ("navigate_to", "walk_steps", "walk_to")

    @staticmethod
    def _is_manipulating(task: Any) -> bool:
        return task.skill_type in ("whole_body_grasp", "bi_manual_carry", "place_object", "handover")
