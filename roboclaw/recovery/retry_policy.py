"""Retry policy with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


class RetryPolicy:
    """Configurable retry policy for skill execution.

    Supports exponential backoff with jitter, max retries, and
    skill-specific overrides.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_s: float = 0.5,
        max_delay_s: float = 30.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter

        # Skill-specific overrides
        self._skill_overrides: dict[str, dict[str, Any]] = {}

    def set_skill_policy(self, skill_type: str, max_retries: int, base_delay_s: float) -> None:
        """Override retry policy for a specific skill type."""
        self._skill_overrides[skill_type] = {
            "max_retries": max_retries,
            "base_delay_s": base_delay_s,
        }

    def get_max_retries(self, skill_type: str = "default") -> int:
        """Get max retries for a skill."""
        override = self._skill_overrides.get(skill_type, {})
        return override.get("max_retries", self.max_retries)

    def get_delay(self, attempt: int, skill_type: str = "default") -> float:
        """Calculate delay for a given retry attempt.

        Uses exponential backoff: base_delay * multiplier^attempt
        With optional random jitter: delay * random(0.75, 1.25)
        """
        override = self._skill_overrides.get(skill_type, {})
        base = override.get("base_delay_s", self.base_delay_s)

        delay = base * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay_s)

        if self.jitter:
            delay *= random.uniform(0.75, 1.25)

        return delay

    async def wait_before_retry(self, attempt: int, skill_type: str = "default") -> None:
        """Wait for the calculated delay before retrying."""
        delay = self.get_delay(attempt, skill_type)
        logger.info(f"Retry attempt {attempt} for '{skill_type}' — waiting {delay:.2f}s")
        await asyncio.sleep(delay)
