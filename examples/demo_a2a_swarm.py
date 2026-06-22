#!/usr/bin/env python3
"""Multi-agent A2A collaboration demo.

Simulates two humanoid robots collaborating:
- Robot A (living room): tasked with "bring me a drink from the kitchen"
- Robot B (kitchen): receives delegated sub-task "pick up the cup"

Demonstrates: AgentCard discovery, task delegation, result retrieval.
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("demo_a2a")


async def main():
    logger.info("=" * 60)
    logger.info("  RoboClaw Multi-Agent A2A Collaboration Demo")
    logger.info("=" * 60)

    # --- 1. Set up agents ---
    logger.info("\n[1] Setting up agents...")

    from roboclaw.a2a.client import A2AClient
    from roboclaw.a2a.models import AgentCard, AgentCapability
    from roboclaw.a2a.negotiation import NegotiationEngine
    from roboclaw.a2a.registry import AgentRegistry
    from roboclaw.a2a.server import A2AServer

    # Agent A — in the living room
    agent_a_card = AgentCard(
        name="Humanoid A (Living Room)",
        description="Humanoid robot stationed in the living room",
        url="http://localhost:8001",
        capabilities=[
            AgentCapability(capability_id="bipedal_locomotion", description="Can walk between rooms"),
            AgentCapability(capability_id="dual_arm_manipulation", description="Can grasp and carry objects"),
            AgentCapability(capability_id="navigation", description="Room-level navigation"),
        ],
    )
    server_a = A2AServer(agent_card=agent_a_card.model_dump())

    # Agent B — already in the kitchen
    agent_b_card = AgentCard(
        name="Humanoid B (Kitchen)",
        description="Humanoid robot stationed in the kitchen",
        url="http://localhost:8002",
        capabilities=[
            AgentCapability(capability_id="dual_arm_manipulation", description="Can grasp and manipulate objects"),
            AgentCapability(capability_id="scene_understanding", description="Can detect and localize objects"),
        ],
    )
    server_b = A2AServer(agent_card=agent_b_card.model_dump())

    logger.info(f"    Agent A: {agent_a_card.name}")
    logger.info(f"    Agent B: {agent_b_card.name}")

    # --- 2. Register agents ---
    logger.info("\n[2] Registering agents in the directory...")

    registry = AgentRegistry()
    registry.register("agent_a", agent_a_card.model_dump(), "http://localhost:8001")
    registry.register("agent_b", agent_b_card.model_dump(), "http://localhost:8002")
    logger.info(f"    Registry has {len(registry)} agents")

    # --- 3. Find best agent for sub-task ---
    logger.info("\n[3] Task: 'bring me a drink from the kitchen'")
    logger.info("    Agent A is in the living room, needs help from kitchen agent...")

    negotiation = NegotiationEngine(registry=registry)
    client = A2AClient()

    # Agent A delegates "pick up the cup" to Agent B
    best_agent = await negotiation.find_best_agent(
        task_description="pick up the cup from the kitchen counter",
        required_capabilities=["dual_arm_manipulation", "scene_understanding"],
    )

    if best_agent:
        logger.info(f"    Best agent found: {best_agent['name']}")
    else:
        logger.warning("    No suitable agent found!")
        return

    # --- 4. Delegate the task ---
    logger.info("\n[4] Delegating sub-task 'pick up the cup' to Agent B...")

    task_id = await client.submit_task(
        agent_url=best_agent["url"],
        task_description="Pick up the cup from the kitchen counter using right arm",
    )
    logger.info(f"    Task submitted: {task_id}")

    # Agent B processes the task
    await server_b.accept_task({
        "task_id": task_id,
        "description": "Pick up the cup from the kitchen counter using right arm",
        "origin_agent": "agent_a",
    })
    await server_b.update_task_status(task_id, "completed", {"grasped": True, "object": "cup"})

    # --- 5. Retrieve result ---
    logger.info("\n[5] Retrieving result from Agent B...")

    result = await client.get_task_result(task_id)
    if result:
        logger.info(f"    Task status: {result.get('status')}")
        logger.info(f"    Result: {result.get('result', {})}")

    # --- 6. Summary ---
    logger.info("\n[6] Collaboration Summary:")
    logger.info("    Agent A (Living Room): 'bring me a drink'")
    logger.info("      ├── Delegated: 'pick up cup' → Agent B (Kitchen)")
    logger.info("      ├── Agent B completed: grasped cup ✓")
    logger.info("      └── Agent A then navigates to kitchen for handover")
    logger.info("\n    Multi-agent collaboration successful! ✓")

    logger.info("\n" + "=" * 60)
    logger.info("  Multi-Agent Demo Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
