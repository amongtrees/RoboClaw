"""Bridge between MuJoCo sensor data and the SafetyMonitor.

The ``MujocoStateProvider`` adapts the snapshot dict produced by
``MujocoEnvironment.get_snapshot()`` into the form expected by
``SafetyMonitor._check_safety()``.
"""

from __future__ import annotations

import logging
from typing import Any

from roboclaw.sim.mujoco_env import MujocoEnvironment

logger = logging.getLogger(__name__)


class MujocoStateProvider:
    """Provides live simulation state in SafetyMonitor-compatible format.

    Wraps a ``MujocoEnvironment`` and translates its snapshot into
    a dict with the keys the SafetyMonitor checks:

    * ``joint_positions`` — {joint_name: rad}
    * ``zmp`` — (zx, zy) tuple
    * ``left_arm_ft`` / ``right_arm_ft`` — force/torque dicts
    * ``com_position`` — [x, y, z]
    * ``contact_forces`` — list of contact dicts

    Parameters
    ----------
    env:
        The shared MuJoCo environment to read from.
    """

    def __init__(self, env: MujocoEnvironment) -> None:
        self._env = env

    async def get_state(self) -> dict[str, Any]:
        """Return the latest simulation state in SafetyMonitor format."""
        snapshot = self._env.get_snapshot()

        if not snapshot:
            # No step executed yet — return minimal safe state
            return {
                "joint_positions": {},
                "zmp": (0.0, 0.0),
                "left_arm_ft": {"force_xyz": (0.0, 0.0, 0.0)},
                "right_arm_ft": {"force_xyz": (0.0, 0.0, 0.0)},
                "com_position": [0.0, 0.0, 1.0],
                "contact_forces": [],
                "support_phase": "double",
            }

        # Extract arm force data from contact forces
        # In a real model with FT sensors, this would use actual sensor data.
        # For now, we approximate from contact forces at the hand geoms.
        left_arm_ft = self._extract_limb_force(snapshot, "left")
        right_arm_ft = self._extract_limb_force(snapshot, "right")

        zmp = snapshot.get("zmp", [0.0, 0.0])
        if isinstance(zmp, list) and len(zmp) >= 2:
            zmp_tuple = (float(zmp[0]), float(zmp[1]))
        else:
            zmp_tuple = (0.0, 0.0)

        # Determine support phase from foot contacts
        contact_forces = snapshot.get("contact_forces", [])
        left_contact = any(
            "left_foot" in str(c.get("geom1", "")) or "left_foot" in str(c.get("geom2", ""))
            for c in contact_forces
        )
        right_contact = any(
            "right_foot" in str(c.get("geom1", "")) or "right_foot" in str(c.get("geom2", ""))
            for c in contact_forces
        )
        if left_contact and right_contact:
            support_phase = "double"
        elif left_contact:
            support_phase = "left"
        elif right_contact:
            support_phase = "right"
        else:
            support_phase = "flight"

        return {
            "joint_positions": snapshot.get("joint_positions", {}),
            "joint_velocities": snapshot.get("joint_velocities", {}),
            "zmp": zmp_tuple,
            "left_arm_ft": left_arm_ft,
            "right_arm_ft": right_arm_ft,
            "com_position": snapshot.get("com_position", [0.0, 0.0, 1.0]),
            "contact_forces": contact_forces,
            "total_contact_force": snapshot.get("total_contact_force", 0.0),
            "support_phase": support_phase,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_limb_force(snapshot: dict[str, Any], side: str) -> dict[str, Any]:
        """Extract approximate force on an arm from contact data.

        In a production model, this would use FT sensor data directly.
        """
        contact_forces = snapshot.get("contact_forces", [])
        total_force = [0.0, 0.0, 0.0]

        for c in contact_forces:
            # Check if this contact involves the target hand
            g1 = str(c.get("geom1", ""))
            g2 = str(c.get("geom2", ""))
            if f"{side}_hand" in g1 or f"{side}_hand" in g2:
                force = c.get("force", [0.0, 0.0, 0.0])
                total_force[0] += force[0]
                total_force[1] += force[1]
                total_force[2] += force[2]

        return {"force_xyz": tuple(total_force)}
