"""ModelRouter — dispatches sub-tasks to the appropriate model client.

This is the central switchboard inserted between ``plan_node`` and
``act_node``.  For each executable SubTask it:

1. Classifies the sub-task into a ``ModelType`` domain (VLN / VLA /
   World Model / LLM / NONE).
2. Routes the sub-task to the corresponding model client.
3. If the model is unavailable, falls back to the in-process simulated
   executor via ``SkillLibrary`` — graceful degradation is a core
   design principle.
"""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.core.config import ModelClientConfig
from roboclaw.core.types import ModelType, model_type_for_skill
from roboclaw.models.clients import (
    AbstractModelClient,
    ModelInferenceRequest,
    ModelInferenceResponse,
)
from roboclaw.models.task import SubTask

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes sub-tasks to VLN / VLA / World Model clients.

    Instantiated once at application startup and shared across all
    agent graph invocations.  Thread-safe for concurrent graph runs
    because each client's ``httpx.AsyncClient`` is itself async-safe.
    """

    def __init__(
        self,
        vln_client: AbstractModelClient | None = None,
        vla_client: AbstractModelClient | None = None,
        world_model_client: AbstractModelClient | None = None,
        skill_library: Any = None,
    ) -> None:
        self._clients: dict[str, AbstractModelClient] = {}
        if vln_client is not None:
            self._clients[ModelType.VLN] = vln_client
        if vla_client is not None:
            self._clients[ModelType.VLA] = vla_client
        if world_model_client is not None:
            self._clients[ModelType.WORLD_MODEL] = world_model_client
        self._skill_library = skill_library

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_subtask(self, subtask: SubTask | dict[str, Any]) -> str:
        """Return the ``ModelType`` that should handle *subtask*.

        Resolution order:
        1. Explicit ``_model_type`` field in the subtask's parameters
           (set by the TaskDecomposer / VLM planner).
        2. SkillType → ModelType mapping (``model_type_for_skill()``).
        3. Default: ``ModelType.LLM``.
        """
        params: dict[str, Any]
        if isinstance(subtask, SubTask):
            params = subtask.parameters
            skill_type = subtask.skill_type
        else:
            params = subtask.get("parameters", {})
            skill_type = subtask.get("skill_type", "")

        # Explicit planner annotation wins
        if "_model_type" in params:
            return params["_model_type"]

        # Fall back to the static skill→model lookup table
        return model_type_for_skill(skill_type)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def route(
        self,
        subtask: SubTask,
        perception: Any | None = None,
        world_state: dict[str, Any] | None = None,
        robot_state: dict[str, Any] | None = None,
    ) -> ModelInferenceResponse:
        """Route a sub-task to the correct model and return the inference result.

        If the target model client is unavailable (or not configured) the
        method transparently falls back to ``SkillLibrary.execute_skill()``
        so the agent loop never breaks.
        """
        model_type = self.classify_subtask(subtask)

        # Resolve SubTask fields whether we have a SubTask model or dict
        if isinstance(subtask, SubTask):
            st_id = subtask.sub_task_id
            skill_type = subtask.skill_type
            parameters = subtask.parameters
        else:
            st_id = subtask.get("sub_task_id", "")
            skill_type = subtask.get("skill_type", "")
            parameters = subtask.get("parameters", {})

        client = self._clients.get(model_type)

        # --- Try model client ---
        if client is not None:
            try:
                if await client.health_check():
                    request = self._build_request(
                        model_type=model_type,
                        sub_task_id=st_id,
                        skill_type=skill_type,
                        parameters=parameters,
                        perception=perception,
                        world_state=world_state or {},
                        robot_state=robot_state or {},
                    )
                    result = await client.infer(request)
                    if result.is_success:
                        logger.info("Model '%s' succeeded for sub-task '%s' (%.0fms)",
                                    model_type, st_id, result.inference_time_ms)
                        return result
                    logger.warning("Model '%s' returned failure for sub-task '%s': %s",
                                   model_type, st_id, result.error)
            except Exception as exc:
                logger.error("Model '%s' client error for sub-task '%s': %s",
                             model_type, st_id, exc)

        # --- Fallback ---
        logger.info("Falling back to simulated executor for sub-task '%s' (model_type=%s)",
                    st_id, model_type)
        return await self._fallback_execute(st_id, skill_type, parameters,
                                            world_state or {})

    async def _fallback_execute(
        self,
        sub_task_id: str,
        skill_type: str,
        parameters: dict[str, Any],
        world_state: dict[str, Any],
    ) -> ModelInferenceResponse:
        """Execute via SkillLibrary's simulated executor."""
        if self._skill_library is not None:
            from roboclaw.action.base import ActionResultStatus

            result = await self._skill_library.execute_skill(
                skill_type, parameters, world_state,
            )
            return ModelInferenceResponse(
                request_id=f"fallback_{sub_task_id}",
                status="success" if result.status == ActionResultStatus.SUCCESS else "failure",
                confidence=0.5,
                inference_time_ms=0.0,
                metadata={"fallback": True, "simulated": True,
                          "result": result.actual_outcome},
            )

        return ModelInferenceResponse(
            request_id=f"fallback_{sub_task_id}",
            status="success",  # graceful degradation — always return success
            confidence=0.3,
            inference_time_ms=0.0,
            metadata={
                "fallback": True,
                "simulated": True,
                "note": "No skill library available — returning stub success",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_request(
        model_type: str,
        sub_task_id: str,
        skill_type: str,
        parameters: dict[str, Any],
        perception: Any | None,
        world_state: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> ModelInferenceRequest:
        """Assemble a ModelInferenceRequest from the current agent context."""
        goal = world_state.get("task_goal", "")

        return ModelInferenceRequest(
            model_type=model_type,
            sub_task_id=sub_task_id,
            skill_type=skill_type,
            parameters=parameters,
            goal=goal,
            instruction=parameters.get("_instruction", ""),
            robot_state=robot_state,
            world_state=world_state,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def registered_model_types(self) -> list[str]:
        return list(self._clients.keys())

    async def close(self) -> None:
        for client in self._clients.values():
            if hasattr(client, "close"):
                await client.close()
