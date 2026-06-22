#!/usr/bin/env python3
"""Demo of the RoboClaw failure recovery subsystem.

Injects a simulated failure and demonstrates the chain-of-responsibility
recovery: SelfDiagnose → RePlan → SafeStop → HumanHelp.
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("demo_recovery")


async def main():
    logger.info("=" * 60)
    logger.info("  RoboClaw Failure Recovery Demo")
    logger.info("=" * 60)

    # --- 1. Set up recovery chain ---
    logger.info("\n[1] Building recovery chain...")

    from roboclaw.recovery.recovery_chain import RecoveryChain
    from roboclaw.recovery.strategies.human_help import HumanHelpStrategy
    from roboclaw.recovery.strategies.replan import RePlanStrategy
    from roboclaw.recovery.strategies.retry_skill import RetrySkillStrategy
    from roboclaw.recovery.strategies.safe_stop import SafeStopStrategy
    from roboclaw.recovery.strategies.self_diagnose import SelfDiagnoseStrategy

    chain = RecoveryChain()
    chain.add_strategy(SelfDiagnoseStrategy())
    chain.add_strategy(RePlanStrategy())
    chain.add_strategy(RetrySkillStrategy())
    chain.add_strategy(SafeStopStrategy())
    chain.add_strategy(HumanHelpStrategy())  # Safety net — always last

    logger.info(f"    Recovery chain built with {chain.strategy_count} strategies:")
    for i, s in enumerate(chain._strategies):
        logger.info(f"      {i+1}. {s.__class__.__name__}")

    # --- 2. Simulate different failure scenarios ---
    from unittest.mock import MagicMock

    scenarios = [
        {
            "name": "Communication timeout",
            "error": {"type": "communication", "detail": "LiDAR driver not responding"},
            "expected": "self_diagnose",
        },
        {
            "name": "Object not found",
            "error": {"type": "perception", "detail": "Cup moved from expected position"},
            "expected": "self_diagnose or replan",
        },
        {
            "name": "Grasp slipped",
            "error": {"type": "dynamic", "detail": "Object slipped during grasp"},
            "expected": "replan",
        },
        {
            "name": "Joint limit reached",
            "error": {"type": "kinematic", "detail": "Shoulder pitch at limit"},
            "expected": "replan or retry_skill",
        },
        {
            "name": "Safety violation",
            "error": {"type": "safety", "detail": "Force limit exceeded on right arm"},
            "expected": "safe_stop",
        },
        {
            "name": "Motor fault",
            "error": {"type": "hardware", "detail": "Left knee motor overheat"},
            "expected": "safe_stop → human_help",
        },
        {
            "name": "Unknown catastrophic failure",
            "error": {"type": "unknown", "detail": "Robot fell over"},
            "expected": "human_help",
        },
    ]

    logger.info(f"\n[2] Testing {len(scenarios)} failure scenarios...")
    logger.info("-" * 40)

    mock_state = MagicMock()
    mock_state.robot_id = "h1_demo_001"
    mock_state.recovery_attempts = 1
    mock_state.task_spec = {"goal": "pick up the cup"}
    mock_state.current_subtask = {
        "sub_task_id": "grasp_cup",
        "skill_type": "whole_body_grasp",
        "parameters": {"object": "cup", "arm": "right"},
    }

    for scenario in scenarios:
        logger.info(f"\n  Scenario: {scenario['name']}")
        logger.info(f"    Error: {scenario['error']['type']} — {scenario['error']['detail']}")

        result = await chain.attempt_recovery(scenario["error"], mock_state)

        logger.info(f"    Result: strategy='{result.strategy_used}', success={result.success}")
        if result.human_message:
            logger.info(f"    Human message: {result.human_message[:100]}...")
        if result.modified_subtask:
            logger.info(f"    Modified subtask params: {result.modified_subtask.get('parameters', {})}")

    logger.info("\n" + "-" * 40)
    logger.info("\n[3] All recovery scenarios tested!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
