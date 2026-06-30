"""SensorFusion — bridges raw sensor data into LangGraph-compatible state updates.

When a ``MujocoEnvironment`` (or future sensor backend) is attached, ``fuse()``
reads the latest snapshot and produces a dict of ``AgentGraphState`` field
updates.  Without a backend, it falls back to minimal stub data so the agent
loop continues uninterrupted.
"""

from __future__ import annotations

import logging
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np

from roboclaw.orchestration.state_schema import AgentGraphState, PerceptionData

if TYPE_CHECKING:
    from roboclaw.sim.mujoco_env import MujocoEnvironment

logger = logging.getLogger(__name__)


class SensorFusion:
    """Fuses multi-modal sensor data into AgentGraphState partial updates.

    Parameters
    ----------
    env:
        A ``MujocoEnvironment`` whose ``get_snapshot()`` provides the raw
        sensor data.  When ``None`` (no simulation), ``fuse()`` returns
        stub perception data.
    """

    def __init__(self, env: MujocoEnvironment | None = None) -> None:
        self._env = env

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fuse(self, state: AgentGraphState) -> dict[str, Any]:
        """Produce a dict of ``AgentGraphState`` partial updates.

        Reads the latest snapshot from the attached environment (if any)
        and maps it to LangGraph-compatible state fields.  When no
        environment is attached or the snapshot is empty, returns stub data.
        """
        if self._env is not None:
            snapshot = self._env.get_snapshot()
            if snapshot:
                try:
                    return self._process_snapshot(snapshot, state)
                except Exception:
                    logger.exception("SensorFusion: snapshot processing failed, falling back to stub")
            else:
                logger.debug("SensorFusion: empty snapshot, using stub")

        return self._build_stub_update(state)

    # ------------------------------------------------------------------
    # Snapshot processing
    # ------------------------------------------------------------------

    def _process_snapshot(
        self,
        snapshot: dict[str, Any],
        state: AgentGraphState,
    ) -> dict[str, Any]:
        """Convert a raw MuJoCo snapshot into AgentGraphState field updates.

        The snapshot dict (from ``MujocoEnvironment._build_snapshot()``)
        contains joint states, IMU, body poses, contacts, COM, and ZMP.
        We map these into the corresponding ``AgentGraphState`` fields
        while also building a rich ``PerceptionData`` object.
        """
        joint_positions: dict[str, float] = {
            k: float(v) for k, v in snapshot.get("joint_positions", {}).items()
        }
        joint_velocities: dict[str, float] = {
            k: float(v) for k, v in snapshot.get("joint_velocities", {}).items()
        }
        imu_data: dict[str, Any] = _ensure_native(snapshot.get("imu", {}))
        body_poses: dict[str, Any] = _ensure_native(snapshot.get("body_poses", {}))
        contact_forces: list[dict[str, Any]] = _ensure_native(
            snapshot.get("contact_forces", [])
        )
        com_position: list[float] = [float(x) for x in snapshot.get("com_position", [])]
        zmp_position: list[float] = [float(x) for x in snapshot.get("zmp", [])]
        total_force = float(snapshot.get("total_contact_force", 0.0))

        # Assemble proprioceptive state bundle
        proprioceptive_state = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "imu": imu_data,
            "com_position": com_position,
            "zmp": zmp_position,
            "total_contact_force": total_force,
        }

        # Build scene graph from body poses
        scene_graph = self._build_scene_graph(body_poses, state.world_objects)

        # Build PerceptionData with sensor-derived information
        detected_objects = list(state.world_objects)  # preserve known objects
        # Augment with body pose info as "objects" the robot knows about itself
        for body_name, pose in body_poses.items():
            detected_objects.append({
                "id": f"body_{body_name}",
                "type": "robot_body_part",
                "name": body_name,
                "position": pose.get("position", []),
                "confidence": 1.0,
            })

        perception = PerceptionData(
            timestamp=snapshot.get("time", time()),
            detected_objects=detected_objects,
            scene_graph=scene_graph,
        )

        # Derive robot_pose from torso/pelvis (whichever is available)
        robot_pose = {}
        _torso_candidates = ("torso", "torso_link", "pelvis")
        torso_name = next((n for n in _torso_candidates if n in body_poses), None)
        if torso_name:
            robot_pose = {
                "position": body_poses[torso_name].get("position", []),
                "rotation": body_poses[torso_name].get("rotation", []),
                "source": f"mujoco_body_pose:{torso_name}",
            }

        logger.debug(
            "SensorFusion: %d joints, %d bodies, %d contacts, COM=[%.2f,%.2f,%.2f]",
            len(joint_positions),
            len(body_poses),
            len(contact_forces),
            com_position[0] if len(com_position) > 0 else 0.0,
            com_position[1] if len(com_position) > 1 else 0.0,
            com_position[2] if len(com_position) > 2 else 0.0,
        )

        return {
            "perception": perception,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "imu_data": imu_data,
            "body_poses": body_poses,
            "contact_forces": contact_forces,
            "com_position": com_position,
            "zmp_position": zmp_position,
            "proprioceptive_state": proprioceptive_state,
            "robot_pose": robot_pose,
        }

    # ------------------------------------------------------------------
    # Stub fallback
    # ------------------------------------------------------------------

    def _build_stub_update(self, state: AgentGraphState) -> dict[str, Any]:
        """Return minimal stub perception data (mirrors the original
        ``perceive_node`` stub behavior when no sensor backend is available)."""
        perception = PerceptionData(
            timestamp=time(),
            detected_objects=state.world_objects,
        )
        return {
            "perception": perception,
            # Keep existing sensor fields as-is (empty defaults) so the
            # agent loop continues with whatever was previously populated.
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scene_graph(
        body_poses: dict[str, Any],
        world_objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a minimal scene graph from body poses and known objects.

        In a full implementation this would run object detection / segmentation.
        For now, it represents robot body parts and known world objects as
        nodes in a spatial graph.
        """
        nodes: list[dict[str, Any]] = []

        # Robot body parts as scene nodes
        for name, pose in body_poses.items():
            nodes.append({
                "id": f"body_{name}",
                "label": name,
                "type": "robot_part",
                "position": pose.get("position", []),
            })

        # Known world objects
        for obj in world_objects:
            nodes.append({
                "id": obj.get("id", f"obj_{len(nodes)}"),
                "label": obj.get("label", obj.get("name", "unknown")),
                "type": obj.get("type", "object"),
                "position": obj.get("position", []),
            })

        return {
            "nodes": nodes,
            "edges": [],  # relationships could be inferred from proximity
        }


# ---------------------------------------------------------------------------
# Utility: convert numpy scalars to native Python types for JSON/msgpack
# ---------------------------------------------------------------------------


def _ensure_native(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python types.

    LangGraph checkpoints use msgpack/json serializers which cannot
    handle ``numpy.float64``, ``numpy.int64``, etc.
    """
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ensure_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_native(v) for v in obj]
    return obj
