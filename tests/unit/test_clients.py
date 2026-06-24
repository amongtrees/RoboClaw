"""Unit tests for VLN, VLA, and World Model API clients."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from roboclaw.clients.vla_client import VLAClient
from roboclaw.clients.vln_client import VLNClient
from roboclaw.clients.world_model_client import WorldModelClient
from roboclaw.core.config import ModelClientConfig
from roboclaw.core.types import ModelType
from roboclaw.models.clients import ModelInferenceRequest, ModelInferenceResponse


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vln_config() -> ModelClientConfig:
    return ModelClientConfig(
        model_type="vln",
        model_name="Qwen-RobotNav-8B",
        base_url="http://localhost:8002",
        api_path="/v1/navigate",
        timeout_sec=5.0,
        max_retries=2,
        enabled=True,
    )


@pytest.fixture
def vla_config() -> ModelClientConfig:
    return ModelClientConfig(
        model_type="vla",
        model_name="GR00T-N1-2B",
        base_url="http://localhost:8003",
        api_path="/v1/act",
        timeout_sec=5.0,
        max_retries=2,
        enabled=True,
    )


@pytest.fixture
def wm_config() -> ModelClientConfig:
    return ModelClientConfig(
        model_type="world_model",
        model_name="NVIDIA-Cosmos-Predict-2",
        base_url="http://localhost:8004",
        api_path="/v1/predict",
        timeout_sec=5.0,
        max_retries=2,
        enabled=True,
    )


@pytest.fixture
def nav_request() -> ModelInferenceRequest:
    return ModelInferenceRequest(
        model_type=ModelType.VLN,
        sub_task_id="navigate_to_kitchen",
        skill_type="navigate_to",
        parameters={"room": "kitchen"},
        goal="Go to the kitchen",
        instruction="Navigate to kitchen",
    )


@pytest.fixture
def act_request() -> ModelInferenceRequest:
    return ModelInferenceRequest(
        model_type=ModelType.VLA,
        sub_task_id="grasp_cup",
        skill_type="whole_body_grasp",
        parameters={"object": "cup", "arm": "right"},
        goal="Pick up the cup",
        instruction="Grasp the cup with right arm",
    )


@pytest.fixture
def predict_request() -> ModelInferenceRequest:
    return ModelInferenceRequest(
        model_type=ModelType.WORLD_MODEL,
        sub_task_id="predict_grasp",
        skill_type="whole_body_grasp",
        parameters={"object": "cup"},
        goal="Predict grasp outcome",
        instruction="What happens if I grasp the cup?",
    )


# ---------------------------------------------------------------------------
# VLNClient
# ---------------------------------------------------------------------------


class TestVLNClient:
    def test_model_type(self, vln_config):
        client = VLNClient(vln_config)
        assert client.model_type == ModelType.VLN

    def test_initial_state_not_available(self, vln_config):
        client = VLNClient(vln_config)
        assert client.is_available is False

    @pytest.mark.asyncio
    async def test_health_check_success(self, vln_config):
        client = VLNClient(vln_config)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = await client.health_check()
            assert result is True
            assert client.is_available is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, vln_config):
        client = VLNClient(vln_config)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = await client.health_check()
            assert result is False
            assert client.is_available is False

    @pytest.mark.asyncio
    async def test_infer_success(self, vln_config, nav_request):
        client = VLNClient(vln_config)
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "request_id": nav_request.request_id,
            "status": "success",
            "waypoints": [{"x": 1.0, "y": 0.0, "z": 0.0, "theta": 0.0}],
            "trajectory": [],
            "confidence": 0.90,
            "inference_time_ms": 150.0,
        }
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.infer(nav_request)
            assert result.is_success
            assert len(result.waypoints) == 1
            assert result.confidence == 0.90

    @pytest.mark.asyncio
    async def test_infer_caches_waypoints(self, vln_config, nav_request):
        client = VLNClient(vln_config)
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "request_id": nav_request.request_id,
            "status": "success",
            "waypoints": [{"x": 1.0, "y": 0.0}, {"x": 2.0, "y": 0.5}],
        }
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.infer(nav_request)
            cached = client.pop_waypoints()
            assert len(cached) == 2
            # Second pop returns empty
            assert client.pop_waypoints() == []

    @pytest.mark.asyncio
    async def test_infer_timeout(self, vln_config, nav_request):
        client = VLNClient(vln_config)
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            result = await client.infer(nav_request)
            assert result.status == "failure"
            assert result.error is not None
            assert result.error["type"] == "vln_timeout"

    @pytest.mark.asyncio
    async def test_infer_http_error(self, vln_config, nav_request):
        client = VLNClient(vln_config)
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock(status_code=500)
            mock_response.text = "Internal Server Error"
            mock_post.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_response
            )
            result = await client.infer(nav_request)
            assert result.status == "failure"
            assert result.error["type"] == "vln_http_error"


# ---------------------------------------------------------------------------
# VLAClient
# ---------------------------------------------------------------------------


class TestVLAClient:
    def test_model_type(self, vla_config):
        client = VLAClient(vla_config)
        assert client.model_type == ModelType.VLA

    @pytest.mark.asyncio
    async def test_infer_success(self, vla_config, act_request):
        client = VLAClient(vla_config)
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "request_id": act_request.request_id,
            "status": "success",
            "joint_actions": {"right_shoulder_pitch": [0.5, 0.5]},
            "ee_poses": [{"x": 0.4, "y": 0.0, "z": 0.85}],
            "gripper_command": "close",
            "action_tokens": [1, 3, 5],
            "confidence": 0.88,
            "inference_time_ms": 62.0,
        }
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.infer(act_request)
            assert result.is_success
            assert result.joint_actions == {"right_shoulder_pitch": [0.5, 0.5]}
            assert result.gripper_command == "close"
            assert result.action_tokens == [1, 3, 5]

    @pytest.mark.asyncio
    async def test_infer_timeout(self, vla_config, act_request):
        client = VLAClient(vla_config)
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            result = await client.infer(act_request)
            assert result.status == "failure"
            assert result.error["type"] == "vla_timeout"


# ---------------------------------------------------------------------------
# WorldModelClient
# ---------------------------------------------------------------------------


class TestWorldModelClient:
    def test_model_type(self, wm_config):
        client = WorldModelClient(wm_config)
        assert client.model_type == ModelType.WORLD_MODEL

    def test_api_key_resolution(self, wm_config):
        wm_config.api_key = "${TEST_API_KEY}"
        import os
        os.environ["TEST_API_KEY"] = "secret-123"
        client = WorldModelClient(wm_config)
        assert client._api_key == "secret-123"
        del os.environ["TEST_API_KEY"]

    def test_api_key_plain_text(self, wm_config):
        wm_config.api_key = "plaintext-key"
        client = WorldModelClient(wm_config)
        assert client._api_key == "plaintext-key"

    @pytest.mark.asyncio
    async def test_infer_success(self, wm_config, predict_request):
        client = WorldModelClient(wm_config)
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "request_id": predict_request.request_id,
            "status": "success",
            "predicted_state": {"zmp": [0.01, -0.02]},
            "uncertainty": 0.15,
            "confidence": 0.85,
            "inference_time_ms": 300.0,
        }
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.infer(predict_request)
            assert result.is_success
            assert result.predicted_state == {"zmp": [0.01, -0.02]}
            assert result.uncertainty == 0.15

    @pytest.mark.asyncio
    async def test_infer_timeout(self, wm_config, predict_request):
        client = WorldModelClient(wm_config)
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            result = await client.infer(predict_request)
            assert result.status == "failure"
            assert result.error["type"] == "wm_timeout"


# ---------------------------------------------------------------------------
# ModelInferenceRequest / Response schemas
# ---------------------------------------------------------------------------


class TestModelInferenceSchemas:
    def test_request_defaults(self):
        req = ModelInferenceRequest()
        assert req.request_id.startswith("req_")
        assert req.model_type == ""
        assert req.rgb_images == []
        assert req.robot_state == {}

    def test_request_serialization(self):
        req = ModelInferenceRequest(
            model_type="vla",
            sub_task_id="grasp_1",
            skill_type="whole_body_grasp",
            instruction="Grasp cup",
        )
        data = req.model_dump(mode="json")
        assert data["model_type"] == "vla"
        assert data["sub_task_id"] == "grasp_1"

    def test_response_success_property(self):
        resp = ModelInferenceResponse(request_id="req_1", status="success")
        assert resp.is_success is True
        resp2 = ModelInferenceResponse(request_id="req_2", status="failure")
        assert resp2.is_success is False

    def test_response_defaults(self):
        resp = ModelInferenceResponse(request_id="req_test")
        assert resp.status == "success"
        assert resp.waypoints == []
        assert resp.joint_actions == {}
        assert resp.predicted_frames == []
        assert resp.confidence == 0.0
