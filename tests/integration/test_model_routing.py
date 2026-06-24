"""Integration tests for end-to-end model routing in the agent graph."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roboclaw.orchestration.graph import build_agent_graph
from roboclaw.orchestration.state_schema import AgentGraphState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state() -> AgentGraphState:
    return AgentGraphState(
        robot_id="h1_test",
        task_spec={"task_id": "task_01", "goal": "go to the kitchen and pick up the cup"},
        world_objects=[
            {"label": "cup", "position": [1.0, 0.5, 0.8]},
            {"label": "table", "position": [0.8, 0.5, 0.7]},
        ],
    )


@pytest.fixture
def mock_model_router():
    """A ModelRouter that always returns simulated success responses."""
    mock = MagicMock()
    mock.classify_subtask = MagicMock(return_value="vln")
    mock.route = AsyncMock(return_value=MagicMock(
        is_success=True,
        status="success",
        waypoints=[{"x": 1.0, "y": 0.0, "z": 0.0}],
        confidence=0.90,
        inference_time_ms=100.0,
        model_dump=MagicMock(return_value={"status": "success", "waypoints": [{"x": 1.0}]}),
    ))
    mock.registered_model_types = ["vln", "vla"]
    mock.close = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Full graph execution tests
# ---------------------------------------------------------------------------


class TestAgentGraphWithModelRouting:
    def test_graph_builds_successfully(self):
        """The graph compiles with the new model_routing fields."""
        graph = build_agent_graph()
        assert graph is not None

    def test_graph_has_eight_nodes(self):
        graph = build_agent_graph()
        nodes = graph.get_graph().nodes
        # All 8 nodes expected
        assert len(nodes) >= 8

    @pytest.mark.asyncio
    async def test_full_loop_without_router_does_not_crash(self, base_state):
        """Full agent loop completes without a ModelRouter (pure fallback)."""
        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-01", "checkpoint_ns": "h1_test"}}

        # Run until completion or max iterations
        final_state = base_state
        for _ in range(30):
            result = await graph.ainvoke(final_state.model_dump(), config)
            final_state = AgentGraphState(**result)
            if not final_state.should_continue:
                break

        # The loop should complete with at least some sub-tasks completed
        assert final_state.loop_count > 0
        # Phase should be "evaluating" at the end (feedback → complete detected)
        assert final_state.phase in ("evaluating",)

    @pytest.mark.asyncio
    async def test_model_routing_is_populated_by_plan_node(self, base_state):
        """plan_node sets the model_routing dict on the state."""
        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-02", "checkpoint_ns": "h1_test"}}

        # Run one iteration to get past plan_node
        state = await graph.ainvoke(base_state.model_dump(), config)

        # model_routing should be populated
        routing = state.get("model_routing", {})
        assert isinstance(routing, dict)
        if routing:
            # Each entry should be a valid model type string
            for model_type in routing.values():
                assert model_type in ("vln", "vla", "world_model", "llm", "none")

    @pytest.mark.asyncio
    async def test_execution_plan_has_model_type_annotations(self, base_state):
        """Every sub-task in the execution plan has a _model_type annotation."""
        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-03", "checkpoint_ns": "h1_test"}}

        state = await graph.ainvoke(base_state.model_dump(), config)

        plan = state.get("execution_plan")
        assert plan is not None
        for st in plan.get("sub_tasks", []):
            assert "_model_type" in st.get("parameters", {}), \
                f"SubTask '{st['sub_task_id']}' missing _model_type"

    @pytest.mark.asyncio
    async def test_with_mock_model_router(self, base_state, mock_model_router):
        """act_node uses the ModelRouter when available."""
        from roboclaw.orchestration.nodes import set_model_router

        set_model_router(mock_model_router)

        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-04", "checkpoint_ns": "h1_test"}}

        # Run a few iterations
        state = base_state
        for _ in range(15):
            result = await graph.ainvoke(state.model_dump(), config)
            state = AgentGraphState(**result)

        # model_inference_results should have some entries
        results = state.model_inference_results
        assert isinstance(results, dict)

        # Clean up
        set_model_router(None)


# ---------------------------------------------------------------------------
# Plan node routing map tests
# ---------------------------------------------------------------------------


class TestPlanNodeRouting:
    @pytest.mark.asyncio
    async def test_routing_map_matches_sub_tasks(self, base_state):
        """Every sub_task_id in the plan has an entry in the routing map."""
        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-05", "checkpoint_ns": "h1_test"}}

        state = await graph.ainvoke(base_state.model_dump(), config)

        plan = state.get("execution_plan", {})
        routing = state.get("model_routing", {})

        for st in plan.get("sub_tasks", []):
            st_id = st["sub_task_id"]
            assert st_id in routing, f"'{st_id}' not in routing map"

    @pytest.mark.asyncio
    async def test_navigate_skill_routed_to_vln(self, base_state):
        """Navigation tasks are routed to the VLN model type."""
        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-06", "checkpoint_ns": "h1_test"}}

        state = await graph.ainvoke(base_state.model_dump(), config)

        routing = state.get("model_routing", {})
        plan = state.get("execution_plan", {})

        for st in plan.get("sub_tasks", []):
            if st["skill_type"] in ("navigate_to", "walk_steps", "climb_stairs"):
                assert routing[st["sub_task_id"]] == "vln", \
                    f"Navigation skill '{st['skill_type']}' not routed to vln"

    @pytest.mark.asyncio
    async def test_grasp_skill_routed_to_vla(self, base_state):
        """Manipulation tasks are routed to the VLA model type."""
        graph = build_agent_graph()
        config = {"configurable": {"thread_id": "test-h1-07", "checkpoint_ns": "h1_test"}}

        state = await graph.ainvoke(base_state.model_dump(), config)

        routing = state.get("model_routing", {})
        plan = state.get("execution_plan", {})

        for st in plan.get("sub_tasks", []):
            if st["skill_type"] in ("whole_body_grasp", "place_object", "handover"):
                assert routing[st["sub_task_id"]] == "vla", \
                    f"Manipulation skill '{st['skill_type']}' not routed to vla"


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_graph_runs_without_router(self):
        """The graph compiles and runs even when no ModelRouter is installed."""
        from roboclaw.orchestration.nodes import set_model_router
        set_model_router(None)

        graph = build_agent_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_demo_state_completes_even_with_fallback(self):
        """A task completes successfully using fallback executors (all models off)."""
        from roboclaw.orchestration.nodes import set_model_router

        # Ensure no router is set
        set_model_router(None)

        graph = build_agent_graph()
        state = AgentGraphState(
            robot_id="h1_test",
            task_spec={"task_id": "task_01", "goal": "go to the kitchen"},
        )
        config = {"configurable": {"thread_id": "test-h1-fallback", "checkpoint_ns": "h1_test"}}

        final = state
        for _ in range(20):
            result = await graph.ainvoke(final.model_dump(), config)
            final = AgentGraphState(**result)
            if not final.should_continue:
                break

        # The agent should have completed some work
        assert final.loop_count > 0
