#!/usr/bin/env python3
"""MuJoCo + LangGraph end-to-end demo with standing balance.

Loads the H1 model, keeps it standing with the StandingController,
and runs LangGraph agent cycles that generate real perception data
and execute skills through the MuJoCo physics engine.

Usage:
    conda activate roboclaw
    python -u scripts/run_mujoco_demo.py         # headless, prints perception
    python -u scripts/run_mujoco_demo.py --view  # with MuJoCo viewer window
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="RoboClaw MuJoCo + LangGraph Demo")
parser.add_argument("--view", action="store_true", help="Launch MuJoCo viewer window")
parser.add_argument("--cycles", type=int, default=10, help="Number of agent cycles")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(name)s: %(message)s")
logger = logging.getLogger("demo")

MODEL_PATH = "models/mujoco_menagerie/unitree_h1/scene.xml"


async def _stand_loop(env: Any, controller: Any, steps: int, com_desired: list[float]) -> None:
    """Step the environment with the standing controller for `steps` steps."""
    for _ in range(steps):
        snap = env.get_snapshot()
        ctrl = controller.compute_control(
            joint_positions=snap.get("joint_positions", {}),
            joint_velocities=snap.get("joint_velocities", {}),
            com_position=snap.get("com_position", []),
            com_desired=com_desired,
        )
        await env.step(ctrl)


async def main() -> None:
    from roboclaw.sim.mujoco_env import MujocoEnvironment
    from roboclaw.sim.controllers import StandingController
    from roboclaw.sim.module import SimModule
    from roboclaw.perception.sensor_fusion import SensorFusion
    from roboclaw.orchestration.state_schema import AgentGraphState
    from roboclaw.orchestration.graph import create_agent_runner

    # ---- 1. Create MuJoCo environment ----
    logger.info("Loading H1 model...")
    env = MujocoEnvironment(MODEL_PATH)
    logger.info("H1: %d actuators, %d joints", env.n_actuators, env.n_joints)

    # ---- 2. Standing controller ----
    stand_ctrl = StandingController(
        joint_names=env.actuator_names,
        default_pose=np.zeros(env.n_actuators),  # qpos0 ≈ 0 for all H1 joints
        kp=80.0,   # high gain to compensate for actuator gainprm=1
        kd=10.0,
        com_kp=20.0,
    )
    com_desired = [0.0, 0.0, 1.0]

    # ---- 3. Let the robot settle into standing ----
    logger.info("Settling robot into standing pose (%d steps)...", 200)
    await _stand_loop(env, stand_ctrl, 200, com_desired)

    # ---- 4. Create SimModule + wire executors ----
    sim_module = SimModule.create_for_testing(env)
    if sim_module is None:
        logger.error("Failed to create SimModule")
        return

    # ---- 5. Build LangGraph agent runner ----
    logger.info("Building agent graph...")
    graph, config = await create_agent_runner(
        robot_id="h1_demo",
        sim_module=sim_module,
    )
    logger.info("Agent graph ready. Starting demo loop...")
    print()

    # ---- 6. Demo loop: stand + periodic agent cycles ----
    sf = SensorFusion(env=env)
    state_dict: dict[str, Any] | None = None

    for cycle in range(args.cycles):
        # --- Run one LangGraph cycle ---
        if state_dict is None:
            # Initial state
            initial = AgentGraphState(
                robot_id="h1_demo",
                task_spec={
                    "task_id": f"demo_{cycle}",
                    "robot_id": "h1_demo",
                    "goal": "stand still and observe the environment",
                    "priority": 0,
                },
            )
            input_state = initial.model_dump()
        else:
            input_state = state_dict

        cycle_start = time.time()
        result = await graph.ainvoke(input_state, config)
        cycle_time = time.time() - cycle_start

        # Extract key state transitions
        phase = result.get("phase", "?")
        perception = result.get("perception")
        plan = result.get("execution_plan")
        action = result.get("last_action_result")
        subtask = result.get("current_subtask")
        completed = result.get("completed_subtask_ids", [])
        error = result.get("error")
        robot_pose = result.get("robot_pose", {})
        com = result.get("com_position", [])
        zmp = result.get("zmp_position", [])
        contacts = len(result.get("contact_forces", []))
        summary = result.get("latest_state_summary", "")

        # Print cycle info
        print(f"━━━ Cycle {cycle + 1}/{args.cycles} "
              f"(phase={phase}, {cycle_time:.3f}s) ━━━")
        if robot_pose:
            rp = robot_pose.get("position", [])
            if len(rp) >= 3:
                print(f"  robot:  [{rp[0]:.3f}, {rp[1]:.3f}, {rp[2]:.3f}]")
        if len(com) >= 3:
            print(f"  COM:    [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
        if len(zmp) >= 2:
            print(f"  ZMP:    [{zmp[0]:.4f}, {zmp[1]:.4f}]")
        print(f"  contacts: {contacts}")
        print(f"  summary:  {summary}")

        if plan:
            subtasks = plan.get("sub_tasks", [])
            print(f"  plan: {len(subtasks)} sub-tasks: "
                  f"{[s['sub_task_id'] for s in subtasks]}")
        if subtask:
            print(f"  executing: {subtask.get('sub_task_id', '?')}")
        if completed:
            print(f"  completed: {completed}")
        if action:
            status = action.get("status", "?")
            dur = action.get("duration_sec", 0)
            print(f"  action: {status} ({dur:.3f}s)")
        if error:
            print(f"  ERROR: {error}")
        print()

        # Store for next cycle
        state_dict = result

        # Stand between cycles
        await _stand_loop(env, stand_ctrl, 100, com_desired)

    # ---- Cleanup ----
    await sim_module.close()
    logger.info("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
