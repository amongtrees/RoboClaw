#!/usr/bin/env python3
"""Interactive LLM-driven agent demo with MuJoCo H1 physics.

Launches H1 in the MuJoCo viewer with standing balance, then accepts
natural language commands that are decomposed by an LLM (or template
fallback) into skill sequences and executed in real physics.

Usage::

    # Template-based planning (no LLM needed):
    python scripts/run_agent_demo.py --no-viewer

    # With LLM (any OpenAI-compatible endpoint):
    python scripts/run_agent_demo.py \\
        --llm-base-url http://localhost:8080/v1 \\
        --llm-model gpt-4o

    # Environment variables:
    export OPENAI_API_KEY="sk-..."
    export ROBOCLAW_LLM_BASE_URL="http://localhost:8080/v1"

Example commands to try::

    > 去厨房拿红色的杯子
    > walk forward 5 steps
    > grasp the bottle with right arm
    > look at the door
    > climb 3 stairs up
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_demo")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "models/mujoco_menagerie/unitree_h1/scene.xml"
STANDING_SETTLE_STEPS = 300  # physics steps to settle before accepting commands


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoboClaw — Interactive H1 Agent Demo with MuJoCo Physics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-path", default=MODEL_PATH,
                   help="Path to H1 scene XML")
    p.add_argument("--no-viewer", action="store_true",
                   help="Run headless (no MuJoCo window)")
    p.add_argument("--no-agent", action="store_true",
                   help="Skip LangGraph agent — execute skills directly")
    p.add_argument("--llm-base-url", default=os.environ.get("ROBOCLAW_LLM_BASE_URL", ""),
                   help="OpenAI-compatible API base URL")
    p.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"),
                   help="API key for the LLM endpoint")
    p.add_argument("--llm-model", default="gpt-4o",
                   help="Model name for planning")
    p.add_argument("--kp", type=float, default=150.0,
                   help="Joint PD position gain")
    p.add_argument("--kd", type=float, default=15.0,
                   help="Joint PD velocity damping gain")
    p.add_argument("--band-ratio", type=float, default=0.65,
                   help="Elastic band weight support fraction (0-1)")
    p.add_argument("--band-kp-xy", type=float, default=100.0,
                   help="Horizontal centering stiffness N/m (lower = easier walking)")
    p.add_argument("--com-fb-gain", type=float, default=30.0,
                   help="Ankle COM feedback gain Nm/m (lower = easier walking)")
    p.add_argument("--sync-every", type=int, default=5,
                   help="Sync the viewer every N physics steps "
                        "(lower = smoother, higher = faster). Default 5.")
    p.add_argument("--low-quality", action="store_true",
                   help="Reduce viewer quality for better performance on "
                        "integrated GPUs or WSL2.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Skill names for direct execution (--no-agent mode)
# ---------------------------------------------------------------------------
SKILL_MAP = {
    "navigate": "navigate_to",
    "walk": "walk_steps",
    "grasp": "whole_body_grasp",
    "place": "place_object",
    "handover": "handover",
    "speak": "speak",
    "gaze": "gaze_at",
    "look": "gaze_at",
    "door": "open_door",
    "open": "open_door",
    "climb": "climb_stairs",
    "stairs": "climb_stairs",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    args = parse_args()

    # --- 1. Import MuJoCo ------------------------------------------------
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("MuJoCo not installed. Run: pip install mujoco", file=sys.stderr)
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        print("Run: bash scripts/download_models.sh", file=sys.stderr)
        sys.exit(1)

    # --- 2. Load model and create BalanceController ----------------------
    from roboclaw.sim.balance import BalanceController

    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)

    balance = BalanceController(
        model=model, data=data,
        kp=args.kp, kd=args.kd,
        band_ratio=args.band_ratio,
        band_kp_xy=args.band_kp_xy,
        com_fb_gain=args.com_fb_gain,
    )

    # --- 3. Settle the robot --------------------------------------------
    print(f"\n{'='*60}")
    print("RoboClaw — H1 Interactive Agent Demo")
    print(f"{'='*60}")
    print(f"H1: {model.nbody} bodies, {model.nu} actuators, mass={balance.total_mass:.1f} kg")
    print(f"Band: {balance.band_force_z:.0f} N ({args.band_ratio*100:.0f}% weight)")
    print(f"PD gains: kp={args.kp}, kd={args.kd}")
    print(f"LLM: {'enabled' if args.llm_base_url else 'disabled (template planning)'}")
    if args.llm_base_url:
        print(f"  endpoint: {args.llm_base_url}")
        print(f"  model: {args.llm_model}")
    print(f"Viewer: {'off (headless)' if args.no_viewer else 'on'}")
    print(f"{'='*60}\n")
    print("Settling balance...", end=" ", flush=True)

    balance.settle(steps=STANDING_SETTLE_STEPS)

    com = data.subtree_com[0] if data.subtree_com.shape[0] > 0 else [0, 0, 0]
    print(f"done. pelvis_z={data.xpos[balance.pelvis_id][2]:.3f}, "
          f"COM=[{com[0]:.3f},{com[1]:.3f},{com[2]:.3f}]")

    # --- 4. Create SimModule and wire executors -------------------------
    from roboclaw.sim.module import SimModule
    from roboclaw.action.skill_library import SkillLibrary

    # Create the environment wrapper — share the SAME mjData as the
    # BalanceController and viewer so physics steps are visible.
    from roboclaw.sim.mujoco_env import MujocoEnvironment
    env = MujocoEnvironment._from_model(model, data=data)

    sim_module = SimModule.create_for_testing(env, balance=balance)
    skill_library = SkillLibrary()
    sim_module.wire_executors(skill_library)

    # --- 5. Setup LLM client (if configured) ----------------------------
    llm = None
    if args.llm_base_url:
        from roboclaw.clients.llm_client import OpenAICompatibleClient
        llm = OpenAICompatibleClient(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
        )
        print(f"LLM client ready: {args.llm_model} @ {args.llm_base_url}\n")

    # --- 6. Build agent graph (if not in direct mode) -------------------
    graph = None
    config = None
    if not args.no_agent:
        from roboclaw.orchestration.graph import create_agent_runner
        graph, config = await create_agent_runner(
            robot_id="h1_demo",
            sim_module=sim_module,
            llm=llm,
        )
        print("Agent graph compiled.\n")

    # --- 7. Help text ---------------------------------------------------
    _print_help(args.no_agent)

    # --- 8. Run ---
    if args.no_viewer:
        await _interactive_loop(
            graph, config, env, balance, skill_library,
            args.no_agent, viewer=None,
        )
    else:
        sync_every = args.sync_every
        if args.low_quality:
            sync_every = max(sync_every, 10)
            print("Low-quality mode: reduced viewer sync rate")

        with mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            # Wire viewer → env so physics steps are visible in real time.
            # Higher sync_every = less GPU load, but jerkier motion.
            env.set_viewer(viewer, sync_every=sync_every)
            await _interactive_loop(
                graph, config, env, balance, skill_library,
                args.no_agent, viewer=viewer,
            )

    # --- 9. Cleanup ---
    await sim_module.close()
    print("\nDemo complete. Goodbye!")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
async def _interactive_loop(
    graph, config, env, balance,
    skill_library, no_agent, viewer,
) -> None:
    """Main read-eval-execute loop."""
    running = True

    def _on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    while running:
        try:
            user_input = await _async_input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() in ("help", "h", "?"):
            _print_help(no_agent)
            continue

        # --- Standalone skill execution (--no-agent) -------------------
        if no_agent:
            await _execute_direct(user_input, env, balance, skill_library, viewer)
            continue

        # --- Agent-driven execution ------------------------------------
        await _execute_via_agent(user_input, graph, config, env, balance, viewer)

        # Maintain balance between commands (env.step handles viewer sync)
        if viewer and viewer.is_running():
            for _ in range(150):
                ctrl = balance.step_control(env)
                await env.step(ctrl)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------
async def _execute_via_agent(
    user_input: str, graph, config, env, balance, viewer,
) -> None:
    """Run the LangGraph agent to decompose and execute the task."""
    from roboclaw.orchestration.state_schema import AgentGraphState

    initial_state = AgentGraphState(
        robot_id="h1_demo",
        task_spec={"task_id": str(uuid4().hex[:12]), "goal": user_input},
    )

    print(f"\n🎯 Task: \"{user_input}\"")
    print(f"   Planning...", end=" ", flush=True)

    t_start = time.time()

    try:
        # Run the agent graph — this will:
        # 1. perceive_node → read MuJoCo sensor data via SensorFusion
        # 2. reflect_node → update working memory
        # 3. plan_node → TaskDecomposer (LLM or template) generates sub-tasks
        # 4. act_node → ModelRouter → SkillLibrary → MuJoCo executor
        # 5. feedback_node → evaluate outcome, loop or complete

        final_state = None
        async for event in graph.astream(
            initial_state.model_dump(), config, stream_mode="values",
        ):
            final_state = event

            # Print sub-task progress
            subtask = event.get("current_subtask")
            if subtask:
                skill = subtask.get("skill_type", "?")
                params = subtask.get("parameters", {})
                st_id = subtask.get("sub_task_id", "")
                phase = event.get("phase", "?")
                print(f"\n   [{phase}] {st_id}: {skill}({_fmt_params(params)})", end=" ", flush=True)

            # Show action results
            action_result = event.get("last_action_result")
            if action_result:
                status = action_result.get("status", "?")
                duration = action_result.get("duration_sec", 0)
                outcome = action_result.get("actual_outcome", {})
                cursor = "✅" if status == "success" else "❌"
                print(f"{cursor} ({duration:.1f}s)", end=" ", flush=True)

            # Maintain balance between graph steps (env.step handles viewer sync)
            if viewer and viewer.is_running():
                for _ in range(20):
                    ctrl = balance.step_control(env)
                    await env.step(ctrl)

        elapsed = time.time() - t_start

        if final_state:
            completed = final_state.get("completed_subtask_ids", [])
            error = final_state.get("error")
            if error:
                print(f"\n   ⚠️  Task error: {error.get('detail', str(error))}")
            else:
                print(f"\n   🎉 Completed {len(completed)} sub-tasks in {elapsed:.1f}s")

    except Exception as exc:
        elapsed = time.time() - t_start
        logger.exception("Agent execution failed")
        print(f"\n   ❌ Agent error ({elapsed:.1f}s): {exc}")


# ---------------------------------------------------------------------------
# Direct skill execution (--no-agent)
# ---------------------------------------------------------------------------
async def _execute_direct(
    user_input: str, env, balance, skill_library, viewer,
) -> None:
    """Execute a skill directly without the LangGraph agent."""
    import mujoco

    parts = user_input.lower().split()
    if not parts:
        return

    # Match first word to a skill
    skill_type = None
    for word in parts:
        if word in SKILL_MAP:
            skill_type = SKILL_MAP[word]
            break

    if skill_type is None:
        # Default: try navigate
        skill_type = "navigate_to"

    # Build parameters heuristically
    params: dict = {}
    if skill_type in ("navigate_to",):
        params["room"] = parts[-1] if len(parts) > 1 else "kitchen"
    elif skill_type == "walk_steps":
        params["num_steps"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
        params["direction"] = "forward"
    elif skill_type == "whole_body_grasp":
        params["object"] = parts[-1] if len(parts) > 1 else "cup"
        params["arm"] = "right"
    elif skill_type == "speak":
        params["text"] = user_input
    elif skill_type == "gaze_at":
        params["target"] = parts[-1] if len(parts) > 1 else "table"
    elif skill_type == "climb_stairs":
        params["num_steps"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
        params["direction"] = "up"
    elif skill_type == "open_door":
        params["door"] = parts[-1] if len(parts) > 1 else "main_door"
        params["arm"] = "right"

    print(f"   Executing: {skill_type}({_fmt_params(params)})", end=" ", flush=True)

    t0 = time.time()
    result = await skill_library.execute_skill(skill_type, params)
    elapsed = time.time() - t0

    cursor = "✅" if result.status.value == "success" else "❌"
    outcome = result.actual_outcome or {}
    print(f"{cursor} ({elapsed:.1f}s)")
    if outcome:
        # Print key outcome fields
        keys = [k for k in outcome if k not in ("simulated", "backend", "params")]
        for k in keys[:5]:
            v = outcome[k]
            if isinstance(v, (list, tuple)) and len(v) <= 3:
                print(f"      {k}: {v}")
        if result.error:
            print(f"      error: {result.error}")

    # Maintain balance after execution (env.step handles viewer sync)
    if viewer and viewer.is_running():
        for _ in range(50):
            ctrl = balance.step_control(env)
            await env.step(ctrl)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_help(no_agent: bool) -> None:
    print("""Commands:
  Navigation:  "go to the kitchen", "walk forward 5 steps"
  Grasping:    "grasp the cup with right arm", "pick up the bottle"
  Placing:     "place the cup on the table"
  Handover:    "hand the cup to me"
  Door:        "open the door with right arm"
  Stairs:      "climb 3 stairs up"
  Gaze:        "look at the door"
  Speak:       "say hello"
  Complex:     "go to the kitchen and pick up the red cup"
  Other:       quit | exit | help""")
    if no_agent:
        print("  (Direct execution mode — skills run immediately)")
    print()


def _fmt_params(params: dict) -> str:
    """Format params dict for display."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts[:4])


async def _async_input(prompt: str = "") -> str:
    """Async wrapper around input()."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
