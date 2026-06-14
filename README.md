# RoboClaw

RoboClaw is a local-first robot memory agent prototype inspired by HelloAgents
chapter 8 and ROSClaw's physical timeline design.

The first milestone focuses on memory, not control:

- Working memory keeps the latest robot state in process memory.
- Episodic memory stores task timelines and outcomes in SQLite.
- Raw sensor artifacts stay on disk; the database stores references.
- The agent exposes a small tool-like API that can later be wired to ROS 2,
  FastAPI, MCP, or LangGraph.

## Why This Shape

For edge robots, the memory system should work without cloud services and should
not put high-frequency sensor streams into a vector database. Robot state is
structured data; task history is a timeline. Semantic search is useful, but it
should sit on top of the timeline rather than replace it.

## Current Components

```text
RoboClaw/
  roboclaw/
    models.py            # Dataclasses for robot state and episodes
    working_memory.py    # TTL-backed current robot state store
    episodic_memory.py   # SQLite episode/event/artifact store
    agent.py             # Tool-like facade over both memory layers
  examples/
    demo_robot_memory.py # End-to-end local demo
  tests/
    test_robot_memory.py # Stdlib unittest coverage
```

## Quick Start

From this directory:

```bash
python examples/demo_robot_memory.py
python -m unittest discover -s tests
```

The demo writes a local SQLite database under `RoboClaw/.roboclaw/`.

## Minimal API

```python
from roboclaw import RobotMemoryAgent

agent = RobotMemoryAgent(robot_id="ur5e")

agent.update_current_state(
    frame_id="base_link",
    camera_frame_id="camera_color_optical_frame",
    end_effector_pose={
        "frame_id": "base_link",
        "position": [0.41, -0.12, 0.30],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    },
)

episode_id = agent.start_episode(
    task_type="grasp",
    goal="pick red cube from table",
    skill_name="top_down_grasp",
)

agent.record_episode_event(
    episode_id,
    event_type="force_feedback",
    payload={"normal_force_n": 8.2, "slip_detected": False},
)

agent.finalize_episode(
    episode_id,
    outcome="success",
    summary="Grasp succeeded with stable force feedback.",
)
```

## Next Engineering Steps

1. Add a ROS 2 adapter that subscribes to `tf`, `joint_states`, camera metadata,
   force/torque, and skill execution topics.
2. Add local embedding search as an optional plugin, keeping SQLite FTS as the
   baseline.
3. Add safety-case memory for blocked actions, force threshold violations, and
   workspace boundary events.
4. Expose the agent through FastAPI or MCP once the memory API stabilizes.
