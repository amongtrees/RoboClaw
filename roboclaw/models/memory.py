"""Memory query, entry, and context models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roboclaw.models.episode import EpisodeRecord


class MemoryQuery(BaseModel):
    """A query to the memory system."""

    text: str = ""
    embedding: list[float] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    include_embeddings: bool = False
    hybrid_search: bool = False  # combine vector + text search


class MemoryEntry(BaseModel):
    """A single entry retrieved from long-term memory."""

    entry_id: str
    memory_type: str  # episode | skill | knowledge | failure
    content: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMContext(BaseModel):
    """Assembled context for LLM planning calls."""

    robot_id: str = ""
    task_goal: str = ""
    current_state_summary: str = ""
    current_phase: str = ""
    relevant_episodes: list[EpisodeRecord] = Field(default_factory=list)
    relevant_semantic: list[dict[str, Any]] = Field(default_factory=list)
    relevant_knowledge: list[dict[str, Any]] = Field(default_factory=list)
    spatial_context: dict[str, Any] = Field(default_factory=dict)
    safety_constraints: list[str] = Field(default_factory=list)
    peer_agents: list[dict[str, Any]] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Compile context into a structured prompt string."""
        sections = [f"## Current State\n{self.current_state_summary}"]

        if self.relevant_episodes:
            eps_text = "\n".join(
                f"- [{e.outcome}] {e.goal} ({e.task_type}): {e.summary}"
                for e in self.relevant_episodes[:5]
            )
            sections.append(f"## Relevant Past Episodes\n{eps_text}")

        if self.relevant_semantic:
            sem_text = "\n".join(
                f"- {item.get('summary', item.get('description', str(item)))}"
                for item in self.relevant_semantic[:5]
            )
            sections.append(f"## Relevant Semantic Memories\n{sem_text}")

        if self.relevant_knowledge:
            kn_text = "\n".join(
                f"- {item.get('key', '')}: {item.get('value', '')}"
                for item in self.relevant_knowledge[:10]
            )
            sections.append(f"## Domain Knowledge\n{kn_text}")

        if self.safety_constraints:
            sc_text = "\n".join(f"- {c}" for c in self.safety_constraints)
            sections.append(f"## Safety Constraints\n{sc_text}")

        if self.spatial_context:
            sections.append(f"## Spatial Context\n{self.spatial_context}")

        return "\n\n".join(sections)
