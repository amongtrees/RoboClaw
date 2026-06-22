"""Chain-of-responsibility pattern for failure recovery.

Tries recovery strategies in priority order. The first strategy that
returns can_handle=True is invoked. If it fails, the chain continues
to the next strategy.
"""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.recovery.base import AbstractRecoveryStrategy, RecoveryResult

logger = logging.getLogger(__name__)


class RecoveryChain:
    """Ordered chain of recovery strategies.

    Strategies are tried in the order they are registered. The first
    strategy that can handle the error is invoked. If it fails, the
    next strategy is tried.

    The last strategy in the chain should always be one that can handle
    any error (e.g., safe_stop or human_help) to avoid dead ends.
    """

    def __init__(self) -> None:
        self._strategies: list[AbstractRecoveryStrategy] = []

    def add_strategy(self, strategy: AbstractRecoveryStrategy) -> None:
        """Add a strategy to the end of the chain."""
        self._strategies.append(strategy)

    def insert_strategy(self, index: int, strategy: AbstractRecoveryStrategy) -> None:
        """Insert a strategy at a specific priority position."""
        self._strategies.insert(index, strategy)

    async def attempt_recovery(self, error: dict[str, Any], state: Any) -> RecoveryResult:
        """Try each strategy in order until one succeeds or all are exhausted.

        Args:
            error: Error dict with 'type' and 'detail'.
            state: Current AgentGraphState.

        Returns:
            RecoveryResult from the first successful strategy, or the last
            strategy's result (typically human_help).
        """
        logger.info(f"Starting recovery chain for error: {error.get('type', 'unknown')}")

        for i, strategy in enumerate(self._strategies):
            if not strategy.can_handle(error, state):
                logger.debug(f"Strategy '{strategy.__class__.__name__}' cannot handle this error, skipping")
                continue

            logger.info(f"Attempting recovery with '{strategy.__class__.__name__}' (strategy {i+1}/{len(self._strategies)})")
            try:
                result = await strategy.handle(error, state)
                if result.success:
                    logger.info(f"Recovery successful via '{result.strategy_used}'")
                    return result
                logger.warning(f"Strategy '{result.strategy_used}' failed, trying next")
            except Exception as e:
                logger.error(f"Strategy '{strategy.__class__.__name__}' raised exception: {e}")

        # If all strategies failed or none could handle it, return a failure result
        logger.error("All recovery strategies exhausted")
        return RecoveryResult(
            success=False,
            strategy_used="exhausted",
            human_message="All recovery strategies have been exhausted. Manual intervention required.",
        )

    @property
    def strategy_count(self) -> int:
        return len(self._strategies)
