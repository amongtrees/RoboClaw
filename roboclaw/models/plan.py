"""Execution plan models."""

from __future__ import annotations

from time import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from roboclaw.models.task import SubTask

PlanStatus = Literal["pending", "executing", "completed", "failed", "cancelled"]


class PlanNode(BaseModel):
    """A node in the execution plan tree (for hierarchical plans)."""

    node_id: str
    sub_task: SubTask
    children: list["PlanNode"] = Field(default_factory=list)
    status: PlanStatus = "pending"


class ExecutionPlan(BaseModel):
    """A full execution plan with DAG of sub-tasks and metadata."""

    plan_id: str
    task_id: str = ""
    task_goal: str = ""
    sub_tasks: list[SubTask] = Field(default_factory=list)
    estimated_duration_sec: float = 0.0
    status: PlanStatus = "pending"
    created_at: float = Field(default_factory=time)
    updated_at: float = Field(default_factory=time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def task_ids(self) -> set[str]:
        return {t.sub_task_id for t in self.sub_tasks}

    def next_executable(self, completed_ids: set[str]) -> list[SubTask]:
        """Return all sub-tasks whose preconditions are satisfied."""
        return [
            t
            for t in self.sub_tasks
            if t.sub_task_id not in completed_ids and t.dependencies_satisfied(completed_ids)
        ]

    def is_complete(self, completed_ids: set[str]) -> bool:
        """Check if all sub-tasks have been completed."""
        return self.task_ids.issubset(completed_ids)

    def progress(self, completed_ids: set[str]) -> float:
        """Return completion ratio (0.0 to 1.0)."""
        if not self.sub_tasks:
            return 1.0
        return len(completed_ids & self.task_ids) / len(self.sub_tasks)
