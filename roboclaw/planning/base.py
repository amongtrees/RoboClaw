"""Abstract base class for planners."""

from __future__ import annotations

import abc
from typing import Any

from roboclaw.models.plan import ExecutionPlan


class AbstractPlanner(abc.ABC):
    """Abstract interface for task/motion planners."""

    @abc.abstractmethod
    async def plan(
        self,
        task_spec: dict[str, Any],
        world_state: dict[str, Any],
        memory_context: Any,
    ) -> ExecutionPlan:
        """Generate an execution plan for the given task.

        Args:
            task_spec: Task specification with goal and parameters.
            world_state: Current world state from perception.
            memory_context: Compiled context from ContextCompiler.

        Returns:
            An ExecutionPlan with ordered SubTasks forming a DAG.
        """
        ...
