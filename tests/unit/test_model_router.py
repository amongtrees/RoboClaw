"""Unit tests for ModelRouter classification and routing logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roboclaw.clients.model_router import ModelRouter
from roboclaw.clients.vla_client import VLAClient
from roboclaw.clients.vln_client import VLNClient
from roboclaw.core.config import ModelClientConfig
from roboclaw.core.types import ModelType
from roboclaw.models.task import SubTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vln_client():
    client = MagicMock(spec=VLNClient)
    client.model_type = ModelType.VLN
    client.health_check = AsyncMock(return_value=True)
    client.infer = AsyncMock(return_value=MagicMock(
        is_success=True, status="success", confidence=0.9,
        waypoints=[{"x": 1.0, "y": 0.0}],
    ))
    client.is_available = True
    return client


@pytest.fixture
def mock_vla_client():
    client = MagicMock(spec=VLAClient)
    client.model_type = ModelType.VLA
    client.health_check = AsyncMock(return_value=True)
    client.infer = AsyncMock(return_value=MagicMock(
        is_success=True, status="success", confidence=0.88,
        joint_actions={"shoulder": [0.5]}, gripper_command="close",
    ))
    client.is_available = True
    return client


@pytest.fixture
def router(mock_vln_client, mock_vla_client):
    return ModelRouter(
        vln_client=mock_vln_client,
        vla_client=mock_vla_client,
        world_model_client=None,
    )


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestModelRouterClassification:
    def test_classify_navigation_skill(self, router):
        st = SubTask(
            sub_task_id="nav_1",
            skill_type="navigate_to",
            parameters={"room": "kitchen"},
        )
        assert router.classify_subtask(st) == ModelType.VLN

    def test_classify_walking_skill(self, router):
        st = SubTask(
            sub_task_id="walk_1",
            skill_type="walk_steps",
            parameters={"num_steps": 3},
        )
        assert router.classify_subtask(st) == ModelType.VLN

    def test_classify_climb_stairs(self, router):
        st = SubTask(
            sub_task_id="climb_1",
            skill_type="climb_stairs",
            parameters={"num_steps": 5},
        )
        assert router.classify_subtask(st) == ModelType.VLN

    def test_classify_grasp_skill(self, router):
        st = SubTask(
            sub_task_id="grasp_1",
            skill_type="whole_body_grasp",
            parameters={"object": "cup", "arm": "right"},
        )
        assert router.classify_subtask(st) == ModelType.VLA

    def test_classify_place_skill(self, router):
        st = SubTask(
            sub_task_id="place_1",
            skill_type="place_object",
            parameters={"object": "cup"},
        )
        assert router.classify_subtask(st) == ModelType.VLA

    def test_classify_handover_skill(self, router):
        st = SubTask(
            sub_task_id="handover_1",
            skill_type="handover",
            parameters={"object": "cup", "receiver": "human"},
        )
        assert router.classify_subtask(st) == ModelType.VLA

    def test_classify_open_door(self, router):
        st = SubTask(
            sub_task_id="door_1",
            skill_type="open_door",
            parameters={"door": "front"},
        )
        assert router.classify_subtask(st) == ModelType.VLA

    def test_classify_speak_skill(self, router):
        st = SubTask(
            sub_task_id="speak_1",
            skill_type="speak",
            parameters={"text": "hello"},
        )
        assert router.classify_subtask(st) == ModelType.LLM

    def test_classify_gaze_skill(self, router):
        st = SubTask(
            sub_task_id="gaze_1",
            skill_type="gaze_at",
            parameters={"target": "scene"},
        )
        assert router.classify_subtask(st) == ModelType.LLM

    def test_classify_explicit_model_type_overrides(self, router):
        """Explicit _model_type in parameters takes precedence over skill mapping."""
        st = SubTask(
            sub_task_id="nav_1",
            skill_type="navigate_to",
            parameters={"room": "kitchen", "_model_type": "llm"},  # override!
        )
        assert router.classify_subtask(st) == "llm"

    def test_classify_dict_subtask(self, router):
        st = {"sub_task_id": "nav_1", "skill_type": "navigate_to",
              "parameters": {"room": "kitchen"}}
        assert router.classify_subtask(st) == ModelType.VLN


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestModelRouterRouting:
    @pytest.mark.asyncio
    async def test_route_navigation_to_vln(self, router, mock_vln_client):
        st = SubTask(
            sub_task_id="nav_1",
            skill_type="navigate_to",
            parameters={"room": "kitchen"},
        )
        result = await router.route(st)
        assert result.is_success
        mock_vln_client.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_manipulation_to_vla(self, router, mock_vla_client):
        st = SubTask(
            sub_task_id="grasp_1",
            skill_type="whole_body_grasp",
            parameters={"object": "cup", "arm": "right"},
        )
        result = await router.route(st)
        assert result.is_success
        mock_vla_client.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_falls_back_when_model_unavailable(self, router, mock_vln_client):
        mock_vln_client.health_check = AsyncMock(return_value=False)
        st = SubTask(
            sub_task_id="nav_1",
            skill_type="navigate_to",
            parameters={"room": "kitchen"},
        )
        result = await router.route(st)
        # Should succeed via fallback
        assert result.status == "success"
        assert result.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_route_falls_back_on_model_error(self, router, mock_vla_client):
        mock_vla_client.infer = AsyncMock(side_effect=RuntimeError("GPU out of memory"))
        st = SubTask(
            sub_task_id="grasp_1",
            skill_type="whole_body_grasp",
            parameters={"object": "cup"},
        )
        result = await router.route(st)
        # Should succeed via fallback
        assert result.status == "success"
        assert result.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_route_llm_skill_no_client(self, router):
        """LLM-type skills with no LLM client fall back cleanly."""
        st = SubTask(
            sub_task_id="speak_1",
            skill_type="speak",
            parameters={"text": "hello"},
        )
        result = await router.route(st)
        # Should fallback since there's no LLM client
        assert result.metadata.get("fallback") is True


# ---------------------------------------------------------------------------
# Router lifecycle
# ---------------------------------------------------------------------------


class TestModelRouterLifecycle:
    def test_registered_model_types(self, router):
        types = router.registered_model_types
        assert ModelType.VLN in types
        assert ModelType.VLA in types
        assert ModelType.WORLD_MODEL not in types  # wasn't configured

    def test_empty_router(self):
        router = ModelRouter()
        assert router.registered_model_types == []

    def test_router_construction_with_none_clients(self):
        router = ModelRouter(vln_client=None, vla_client=None)
        assert router.registered_model_types == []
