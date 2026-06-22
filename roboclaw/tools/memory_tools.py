"""LangChain tools wrapping memory CRUD operations for LLM agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_memory_tools(
    episodic_memory: Any = None,
    semantic_memory: Any = None,
    working_memory: Any = None,
) -> list:
    """Create LangChain tools for memory operations.

    Tools provided:
    - recall_episodes: Search past episodes by query text
    - recall_similar: Find semantically similar past tasks
    - remember_fact: Store a fact for the current task
    - get_task_context: Get current task context from working memory

    Returns:
        List of LangChain StructuredTool objects.
    """
    tools = []

    try:
        from langchain_core.tools import tool

        @tool
        async def recall_episodes(query: str, limit: int = 5) -> str:
            """Search past task episodes for relevant experiences. Use this to learn from previous successes and failures."""
            logger.info(f"Tool: recall_episodes(query={query}, limit={limit})")
            if episodic_memory:
                episodes = await episodic_memory.search_episodes(query_text=query, limit=limit)
                if episodes:
                    return "\n".join(
                        f"- [{e.outcome}] {e.goal}: {e.summary}" for e in episodes
                    )
            return "No relevant past episodes found."

        @tool
        async def recall_similar(task_description: str, top_k: int = 5) -> str:
            """Find semantically similar past tasks using vector search. Good for finding analogous situations."""
            logger.info(f"Tool: recall_similar(description={task_description}, top_k={top_k})")
            if semantic_memory:
                # In production: embed the task_description and search Milvus
                results = await semantic_memory.find_similar_episodes([], top_k=top_k)
                if results:
                    return "\n".join(f"- {r.get('metadata', r)}" for r in results)
            return "No semantically similar tasks found."

        @tool
        async def remember_fact(key: str, value: str) -> str:
            """Store a fact or observation for the current task context."""
            logger.info(f"Tool: remember_fact(key={key}, value={value})")
            if working_memory:
                await working_memory.update_state(**{key: value})
            return f"Remembered: {key} = {value}"

        @tool
        async def get_task_context() -> str:
            """Retrieve the current task context, including goal, plan, and recent observations."""
            logger.info("Tool: get_task_context()")
            if working_memory:
                ctx = await working_memory.get_task_context()
                return str(ctx) if ctx else "No task context available."
            return "Task context unavailable."

        tools = [recall_episodes, recall_similar, remember_fact, get_task_context]
    except ImportError:
        logger.warning("LangChain not available; memory tools not created")

    return tools
