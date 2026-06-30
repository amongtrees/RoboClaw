"""Speech executor for HRI communication.

This executor has *no* MuJoCo physics dependency — speaking is a
pure software action.  It prints the speech text to the console
(or streams it via TTS when a TTS engine is configured).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus

logger = logging.getLogger(__name__)


class MujocoSpeechExecutor(AbstractActionExecutor):
    """Executor for the ``speak`` skill — text-to-speech / console output.

    Parameters
    ----------
    tts_engine:
        Optional TTS callable ``async def tts(text: str) -> None``.
        When omitted, text is printed to stdout.
    """

    def __init__(self, tts_engine: Any = None) -> None:
        self._tts = tts_engine
        self._cancelled = False

    # ------------------------------------------------------------------
    # AbstractActionExecutor interface
    # ------------------------------------------------------------------

    async def execute(
        self, sub_task: dict[str, Any], world_state: dict[str, Any]
    ) -> ActionResult:
        self._cancelled = False
        params = sub_task.get("parameters", {})
        sub_task_id = sub_task.get("sub_task_id", "speak")
        text: str = str(params.get("text", ""))

        if not text.strip():
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.SUCCESS,
                duration_sec=0.0,
                actual_outcome={"spoken": True, "text": "", "note": "empty text"},
            )

        volume: float = float(params.get("volume", 0.8))

        try:
            if self._cancelled:
                return ActionResult(
                    sub_task_id=sub_task_id,
                    status=ActionResultStatus.INTERRUPTED,
                    error={"type": "cancelled"},
                )

            if self._tts is not None:
                await self._tts(text)
            else:
                # Console output — the "voice" of the robot
                print(f"\n🤖 H1 says: \"{text}\"\n", flush=True)

            # Small delay to simulate speaking time (~150 words/min)
            word_count = len(text.split())
            speak_duration = max(0.5, word_count * 0.4)
            await asyncio.sleep(min(speak_duration, 3.0))

            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.SUCCESS,
                duration_sec=speak_duration,
                actual_outcome={
                    "spoken": True,
                    "text": text,
                    "volume": volume,
                    "word_count": word_count,
                },
            )

        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled"},
            )

    async def cancel(self) -> None:
        self._cancelled = True
        logger.info("MujocoSpeechExecutor cancelled")

    async def emergency_stop(self) -> None:
        self._cancelled = True
        logger.warning("MujocoSpeechExecutor emergency stop!")
