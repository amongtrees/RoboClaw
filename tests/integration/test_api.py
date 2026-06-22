"""Integration tests for the FastAPI layer."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Create the FastAPI test app."""
    from roboclaw.api.app import create_app
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async HTTP test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestAdminRoutes:
    @pytest.mark.asyncio
    async def test_health(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_readiness(self, client):
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_metrics(self, client):
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200


class TestAgentRoutes:
    @pytest.mark.asyncio
    async def test_register_agent(self, client):
        response = await client.post("/api/v1/agents", json={
            "robot_id": "h1_test_001",
            "robot_model": "h1_unitree",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["robot_id"] == "h1_test_001"
        assert data["status"] == "idle"

    @pytest.mark.asyncio
    async def test_get_agent(self, client):
        # Register first
        await client.post("/api/v1/agents", json={"robot_id": "h1_test_002"})
        response = await client.get("/api/v1/agents/h1_test_002")
        assert response.status_code == 200
        assert response.json()["robot_id"] == "h1_test_002"

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, client):
        response = await client.get("/api/v1/agents/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_set_agent_mode(self, client):
        await client.post("/api/v1/agents", json={"robot_id": "h1_test_003"})
        response = await client.put("/api/v1/agents/h1_test_003/mode", json={"mode": "teleop"})
        assert response.status_code == 200
        assert response.json()["mode"] == "teleop"

    @pytest.mark.asyncio
    async def test_invalid_mode(self, client):
        await client.post("/api/v1/agents", json={"robot_id": "h1_test_004"})
        response = await client.put("/api/v1/agents/h1_test_004/mode", json={"mode": "flying"})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decommission_agent(self, client):
        await client.post("/api/v1/agents", json={"robot_id": "h1_test_005"})
        response = await client.delete("/api/v1/agents/h1_test_005")
        assert response.status_code == 200
        # Verify gone
        response = await client.get("/api/v1/agents/h1_test_005")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_agents(self, client):
        await client.post("/api/v1/agents", json={"robot_id": "agent_a"})
        await client.post("/api/v1/agents", json={"robot_id": "agent_b"})
        response = await client.get("/api/v1/agents")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) >= 2


class TestTaskRoutes:
    @pytest.mark.asyncio
    async def test_submit_task(self, client):
        response = await client.post("/api/v1/tasks", json={
            "robot_id": "h1_test",
            "goal": "go to the kitchen",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["goal"] == "go to the kitchen"
        assert data["status"] == "accepted"
        assert data["task_id"].startswith("task_")

    @pytest.mark.asyncio
    async def test_submit_task_missing_fields(self, client):
        response = await client.post("/api/v1/tasks", json={})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_task(self, client):
        create_resp = await client.post("/api/v1/tasks", json={
            "robot_id": "h1_test",
            "goal": "test",
        })
        task_id = create_resp.json()["task_id"]

        response = await client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_cancel_task(self, client):
        create_resp = await client.post("/api/v1/tasks", json={
            "robot_id": "h1_test",
            "goal": "test",
        })
        task_id = create_resp.json()["task_id"]

        response = await client.delete(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"


class TestA2ARoutes:
    @pytest.mark.asyncio
    async def test_agent_card(self, client):
        response = await client.get("/.well-known/agent.json")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RoboClaw Humanoid Agent"
        assert "capabilities" in data

    @pytest.mark.asyncio
    async def test_a2a_submit_task(self, client):
        response = await client.post("/a2a/tasks", json={
            "description": "Pick up the cup",
            "origin_agent": "agent_a",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
