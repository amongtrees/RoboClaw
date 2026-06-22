#!/usr/bin/env python3
"""End-to-end demo of the RoboClaw agent loop without hardware.

Runs the full Perceive → Reflect → Plan → Act → Feedback loop
using simulated perception and action. Demonstrates:
- Agent graph creation and execution
- Task submission and plan generation
- Multi-step task completion
- Episode recording
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("demo")


async def main():
    logger.info("=" * 60)
    logger.info("  RoboClaw Full Agent Loop Demo")
    logger.info("=" * 60)

    # --- 1. Create the agent graph ---
    logger.info("\n[1] Building the agent graph...")
    from roboclaw.orchestration.graph import create_agent_runner

    robot_id = "h1_demo_001"
    graph, config = await create_agent_runner(robot_id)
    logger.info(f"    Agent runner created for robot '{robot_id}'")

    # --- 2. Define a task ---
    logger.info("\n[2] Submitting task: 'go to the kitchen and pick up the cup'")

    initial_state = {
        "robot_id": robot_id,
        "task_spec": {
            "task_id": "task_demo_001",
            "robot_id": robot_id,
            "goal": "go to the kitchen and pick up the cup",
            "priority": 0,
        },
        "world_objects": [
            {"label": "doorway", "position": [1.0, 0.0, 0.0]},
            {"label": "kitchen_counter", "position": [3.0, 0.5, 1.0]},
            {"label": "cup", "position": [2.9, 0.6, 1.1]},
            {"label": "table", "position": [3.5, 0.0, 0.8]},
        ],
        "max_recovery_attempts": 3,
    }

    # --- 3. Run the agent loop ---
    logger.info("\n[3] Running the agent loop...")
    logger.info("    (Each iteration runs one phase of the agent cycle)")
    logger.info("-" * 40)

    iteration = 0
    final_state = None

    async for event in graph.astream(initial_state, config, stream_mode="values"):
        iteration += 1
        phase = event.get("phase", "unknown")
        loop_count = event.get("loop_count", 0)

        # Print progress
        current = event.get("current_subtask")
        if current:
            logger.info(f"  [{phase.upper()}] Executing: {current.get('sub_task_id')} ({current.get('skill_type')})")
        elif event.get("execution_plan"):
            plan = event["execution_plan"]
            task_count = len(plan.get("sub_tasks", []))
            logger.info(f"  [{phase.upper()}] Plan generated with {task_count} sub-tasks")
        elif phase == "evaluating":
            completed = event.get("completed_subtask_ids", [])
            logger.info(f"  [{phase.upper()}] Completed: {completed}")
        else:
            logger.info(f"  [{phase.upper()}] Loop {loop_count}")

        final_state = event

        # Safety: max iterations
        if iteration > 30:
            logger.warning("  Reached max iterations, stopping demo")
            break

    logger.info("-" * 40)

    # --- 4. Report results ---
    logger.info("\n[4] Results:")
    if final_state:
        completed = final_state.get("completed_subtask_ids", [])
        plan = final_state.get("execution_plan", {})
        total_subtasks = len(plan.get("sub_tasks", []))

        logger.info(f"    Final phase: {final_state.get('phase', 'unknown')}")
        logger.info(f"    Sub-tasks completed: {len(completed)}/{total_subtasks}")
        logger.info(f"    Completed IDs: {completed}")
        logger.info(f"    Total loop iterations: {iteration}")

        if final_state.get("error"):
            logger.warning(f"    Error: {final_state['error']}")
        else:
            logger.info("    Status: ALL TASKS COMPLETED SUCCESSFULLY ✓")

    # --- 5. Episode summary ---
    logger.info("\n[5] Episode Summary:")
    episode_id = final_state.get("episode_id", "N/A") if final_state else "N/A"
    logger.info(f"    Episode ID: {episode_id}")
    logger.info(f"    Robot: {robot_id}")
    logger.info(f"    Task: go to the kitchen and pick up the cup")
    logger.info(f"    Iterations: {iteration}")

    logger.info("\n" + "=" * 60)
    logger.info("  Demo Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
