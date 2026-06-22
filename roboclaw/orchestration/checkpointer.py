"""LangGraph PostgreSQL checkpointer setup.

Provides the PostgresSaver for persisting agent graph state across
process restarts, enabling fault-tolerant long-running tasks.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def create_postgres_checkpointer(conn_string: str) -> Any | None:
    """Create a PostgreSQL-backed LangGraph checkpointer.

    Args:
        conn_string: PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)

    Returns:
        PostgresSaver instance, or None if unavailable.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver.from_conn_string(conn_string)
        await checkpointer.setup()
        logger.info("LangGraph PostgreSQL checkpointer initialized")
        return checkpointer
    except ImportError:
        logger.warning("langgraph-checkpoint-postgres not installed; using in-memory checkpointer")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize PostgreSQL checkpointer: {e}; using in-memory fallback")
        return None


def create_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer for development/testing."""
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Using in-memory LangGraph checkpointer")
    return MemorySaver()
