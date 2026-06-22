"""Task specification and execution plan models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """A high-level task submitted to the robot agent."""

    task_id: str = ""
    robot_id: str = ""
    goal: str = ""
    priority: int = 0  # lower = higher priority
    deadline_sec: float | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubTask(BaseModel):
    """An atomic sub-task within an execution plan, with preconditions forming a DAG."""

    sub_task_id: str
    skill_type: str  # SkillType enum value
    parameters: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)  # sub_task_ids that must complete first
    expected_outcome: dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = 30.0
    priority: int = 0
    retry_policy: str = "default"  # default | never | always_retry

    def dependencies_satisfied(self, completed_ids: set[str]) -> bool:
        """Check if all preconditions have been met."""
        return set(self.preconditions).issubset(completed_ids)


class TaskGraph(BaseModel):
    """DAG of sub-tasks representing the full execution plan."""

    sub_tasks: list[SubTask] = Field(default_factory=list)

    @property
    def task_ids(self) -> set[str]:
        return {t.sub_task_id for t in self.sub_tasks}

    def next_executable(self, completed_ids: set[str]) -> list[SubTask]:
        """Return all sub-tasks whose preconditions are satisfied and not yet completed."""
        return [
            t
            for t in self.sub_tasks
            if t.sub_task_id not in completed_ids and t.dependencies_satisfied(completed_ids)
        ]

    def is_complete(self, completed_ids: set[str]) -> bool:
        """Check if all sub-tasks have been completed."""
        return self.task_ids.issubset(completed_ids)
