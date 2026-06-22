"""Speech/TTS executor stub."""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


class SpeechExecutor(AbstractActionExecutor):
    """Text-to-speech output executor.

    Phase 1 stub — logs the speech text.
    """

    def __init__(self) -> None:
        self._cancelled = False

    async def execute(self, sub_task: dict[str, Any], world_state: dict[str, Any]) -> ActionResult:
        params = sub_task.get("parameters", {})
        text = params.get("text", "")
        volume = params.get("volume", 0.8)

        logger.info(f"TTS: '{text}' (volume={volume})")

        return ActionResult(
            sub_task_id=sub_task.get("sub_task_id", "speak"),
            status=ActionResultStatus.SUCCESS,
            duration_sec=len(text) * 0.05,
            actual_outcome={"spoken": True, "text": text, "simulated": True},
        )

    async def cancel(self) -> None:
        self._cancelled = True

    async def emergency_stop(self) -> None:
        self._cancelled = True
