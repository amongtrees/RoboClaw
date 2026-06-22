"""Unit tests for the recovery subsystem."""

from unittest.mock import MagicMock

import pytest

from roboclaw.recovery.recovery_chain import RecoveryChain
from roboclaw.recovery.retry_policy import RetryPolicy
from roboclaw.recovery.safety_monitor import SafetyMonitor
from roboclaw.recovery.strategies.human_help import HumanHelpStrategy
from roboclaw.recovery.strategies.replan import RePlanStrategy
from roboclaw.recovery.strategies.retry_skill import RetrySkillStrategy
from roboclaw.recovery.strategies.safe_stop import SafeStopStrategy
from roboclaw.recovery.strategies.self_diagnose import SelfDiagnoseStrategy


class TestRetryPolicy:
    def test_default_values(self):
        rp = RetryPolicy()
        assert rp.max_retries == 3
        assert rp.base_delay_s == 0.5
        assert rp.max_delay_s == 30.0

    def test_get_delay_exponential(self):
        rp = RetryPolicy(base_delay_s=1.0, backoff_multiplier=2.0, jitter=False)
        # delay = base * multiplier^(attempt-1)
        assert rp.get_delay(1) == 1.0  # 1.0 * 2^0
        assert rp.get_delay(2) == 2.0  # 1.0 * 2^1
        assert rp.get_delay(3) == 4.0  # 1.0 * 2^2

    def test_get_delay_max_cap(self):
        rp = RetryPolicy(base_delay_s=10.0, max_delay_s=15.0, jitter=False)
        assert rp.get_delay(5) == 15.0  # capped at max

    def test_skill_override(self):
        rp = RetryPolicy()
        rp.set_skill_policy("whole_body_grasp", max_retries=5, base_delay_s=2.0)
        assert rp.get_max_retries("whole_body_grasp") == 5
        assert rp.get_max_retries("walk_steps") == 3  # default

    def test_jitter_adds_variation(self):
        rp = RetryPolicy(base_delay_s=1.0, jitter=True)
        delays = [rp.get_delay(1) for _ in range(20)]
        # With jitter, not all delays should be identical
        assert len(set(round(d, 2) for d in delays)) > 1


class TestRecoveryStrategies:
    @pytest.fixture
    def mock_state(self):
        state = MagicMock()
        state.robot_id = "h1_test"
        state.recovery_attempts = 1
        state.task_spec = {"goal": "test task"}
        state.current_subtask = {
            "sub_task_id": "test_subtask",
            "skill_type": "whole_body_grasp",
            "parameters": {"object": "cup", "arm": "right"},
        }
        return state

    def test_self_diagnose_can_handle(self):
        strategy = SelfDiagnoseStrategy()
        assert strategy.can_handle({"type": "communication"}, None)
        assert strategy.can_handle({"type": "perception"}, None)
        assert not strategy.can_handle({"type": "kinematic"}, None)
        assert not strategy.can_handle({"type": "hardware"}, None)

    def test_replan_can_handle(self):
        strategy = RePlanStrategy()
        assert strategy.can_handle({"type": "kinematic"}, None)
        assert strategy.can_handle({"type": "dynamic"}, None)
        assert not strategy.can_handle({"type": "communication"}, None)

    def test_safe_stop_can_handle(self):
        strategy = SafeStopStrategy()
        assert strategy.can_handle({"type": "safety"}, None)
        assert strategy.can_handle({"type": "hardware"}, None)
        assert not strategy.can_handle({"type": "perception"}, None)

    def test_human_help_always_can_handle(self):
        strategy = HumanHelpStrategy()
        assert strategy.can_handle({"type": "any_error"}, None)
        assert strategy.can_handle({}, None)
        assert strategy.can_handle({"type": "unknown"}, None)

    @pytest.mark.asyncio
    async def test_self_diagnose_handle_communication(self, mock_state):
        strategy = SelfDiagnoseStrategy()
        result = await strategy.handle({"type": "communication"}, mock_state)
        assert result.success is True
        assert result.strategy_used == "self_diagnose"

    @pytest.mark.asyncio
    async def test_human_help_generates_message(self, mock_state):
        strategy = HumanHelpStrategy()
        result = await strategy.handle({"type": "unknown", "detail": "Unexpected error"}, mock_state)
        assert result.success is True  # Escalation to human is a successful handling
        assert result.human_message is not None
        assert "h1_test" in result.human_message

    @pytest.mark.asyncio
    async def test_retry_skill_adjusts_params(self, mock_state):
        strategy = RetrySkillStrategy()
        result = await strategy.handle({"type": "kinematic"}, mock_state)
        assert result.success is True
        assert result.modified_subtask is not None
        # Should have switched arm from right to left for kinematic error
        assert result.modified_subtask["parameters"]["arm"] == "left"


class TestRecoveryChain:
    @pytest.fixture
    def chain(self):
        chain = RecoveryChain()
        chain.add_strategy(SelfDiagnoseStrategy())
        chain.add_strategy(RePlanStrategy())
        chain.add_strategy(SafeStopStrategy())
        chain.add_strategy(HumanHelpStrategy())
        return chain

    @pytest.fixture
    def mock_state(self):
        state = MagicMock()
        state.robot_id = "h1_test"
        state.recovery_attempts = 1
        state.task_spec = {"goal": "test task"}
        state.current_subtask = {"sub_task_id": "st1", "skill_type": "walk", "parameters": {}}
        return state

    @pytest.mark.asyncio
    async def test_chain_uses_first_capable_strategy(self, chain, mock_state):
        # SelfDiagnose can handle communication errors
        result = await chain.attempt_recovery({"type": "communication"}, mock_state)
        assert result.strategy_used == "self_diagnose"

    @pytest.mark.asyncio
    async def test_chain_skips_uncapable_strategies(self, chain, mock_state):
        # SelfDiagnose can't handle kinematic, but RePlan can
        result = await chain.attempt_recovery({"type": "kinematic"}, mock_state)
        assert result.strategy_used == "replan"

    @pytest.mark.asyncio
    async def test_chain_falls_back_to_human_help(self, chain, mock_state):
        # Nothing handles "unknown" except HumanHelp (always handles)
        result = await chain.attempt_recovery({"type": "unknown"}, mock_state)
        assert result.strategy_used == "human_help"

    def test_chain_strategy_count(self, chain):
        assert chain.strategy_count == 4


class TestSafetyMonitor:
    def test_initial_state(self):
        monitor = SafetyMonitor("h1_test")
        assert monitor.is_safe

    def test_point_in_polygon_inside(self):
        polygon = [(-0.1, -0.05), (-0.1, 0.05), (0.05, 0.05), (0.05, -0.05)]
        assert SafetyMonitor._point_in_polygon((0.0, 0.0), polygon)

    def test_point_in_polygon_outside(self):
        polygon = [(-0.1, -0.05), (-0.1, 0.05), (0.05, 0.05), (0.05, -0.05)]
        assert not SafetyMonitor._point_in_polygon((1.0, 0.0), polygon)
