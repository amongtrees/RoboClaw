"""MuJoCo-based World Model for predictive safety lookahead.

The ``MujocoWorldModel`` clones the current simulation state and
rolls out candidate action sequences to predict future states.
It checks safety constraints (joint limits, ZMP, collisions) at
every predicted step, enabling pre-execution validation.

This integrates with the existing World Model infrastructure
(``roboclaw/clients/world_model_client.py`` and
``roboclaw/serve/world_model_server.py``) by providing a local,
in-process alternative to remote GPU-hosted models like Cosmos.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from roboclaw.sim.mujoco_env import MujocoEnvironment, _check_mujoco

logger = logging.getLogger(__name__)


class PredictionResult:
    """Result of a world model rollout.

    Attributes
    ----------
    predicted_state:
        The final snapshot after rolling out the action sequence.
    predicted_frames:
        List of snapshots throughout the rollout (one per action step).
    safety_violations:
        List of violations detected during the rollout.
    uncertainty:
        Heuristic uncertainty (0.0–1.0).  Higher = less certain.
    is_safe:
        True if no safety violations were detected.
    """

    def __init__(
        self,
        predicted_state: dict[str, Any],
        predicted_frames: list[dict[str, Any]],
        safety_violations: list[dict[str, Any]],
        uncertainty: float,
        is_safe: bool,
    ) -> None:
        self.predicted_state = predicted_state
        self.predicted_frames = predicted_frames
        self.safety_violations = safety_violations
        self.uncertainty = uncertainty
        self.is_safe = is_safe


class MujocoWorldModel:
    """Predictive World Model using MuJoCo physics.

    Clones the MuJoCo data state and rolls out action sequences to
    predict future states and detect potential safety violations.

    Parameters
    ----------
    base_env:
        The MuJoCo environment to clone for predictions.
    horizon:
        Maximum rollout steps (clamped per call).
    zmp_polygon:
        Support polygon vertices for ZMP checks.
    joint_limits:
        Per-joint position limits dict.  When empty, uses defaults.
    collision_force_threshold:
        Maximum allowed contact force magnitude (N).
    """

    def __init__(
        self,
        base_env: MujocoEnvironment,
        horizon: int = 50,
        zmp_polygon: list[tuple[float, float]] | None = None,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        collision_force_threshold: float = 50.0,
    ) -> None:
        if not _check_mujoco():
            raise ImportError("MuJoCo is not installed")

        self._base_env = base_env
        self._horizon = horizon
        self._zmp_polygon = zmp_polygon or [
            (-0.10, -0.05), (-0.10, 0.05), (0.05, 0.05), (0.05, -0.05),
        ]
        self._joint_limits = joint_limits or {}
        self._collision_force_threshold = collision_force_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict(
        self,
        action_sequence: list[np.ndarray],
        initial_snapshot: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Roll out an action sequence and predict the outcome.

        Parameters
        ----------
        action_sequence:
            List of control vectors to apply sequentially.
        initial_snapshot:
            Snapshot to start from.  Uses the current env state when None.

        Returns
        -------
        PredictionResult
            Including predicted state, safety violations, and uncertainty.
        """
        import mujoco

        # Clone model + data from the base environment
        model = self._base_env.model
        data = self._base_env.data
        cloned_data = copy.deepcopy(data)

        # Reset to initial snapshot if provided
        if initial_snapshot:
            if "qpos" in initial_snapshot:
                cloned_data.qpos[:] = initial_snapshot["qpos"]
            if "qvel" in initial_snapshot:
                cloned_data.qvel[:] = initial_snapshot["qvel"]
            if "ctrl" in initial_snapshot:
                cloned_data.ctrl[:] = initial_snapshot["ctrl"]

        # Forward kinematics from the initial state
        mujoco.mj_forward(model, cloned_data)

        violations: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        steps = min(len(action_sequence), self._horizon)

        for i in range(steps):
            ctrl = np.asarray(action_sequence[i], dtype=np.float64)
            if ctrl.shape[0] == model.nu:
                cloned_data.ctrl[:] = ctrl
            else:
                cloned_data.ctrl[:] = 0.0

            mujoco.mj_step(model, cloned_data)

            # Build frame snapshot
            frame = self._build_frame(model, cloned_data, i)
            frames.append(frame)

            # Check safety constraints
            step_violations = self._check_constraints(model, cloned_data, i)
            violations.extend(step_violations)

        # Compute uncertainty heuristic
        uncertainty = self._compute_uncertainty(violations, steps, action_sequence)

        predicted_state = frames[-1] if frames else {}

        return PredictionResult(
            predicted_state=predicted_state,
            predicted_frames=frames,
            safety_violations=violations,
            uncertainty=uncertainty,
            is_safe=len(violations) == 0,
        )

    # ------------------------------------------------------------------
    # Safety constraint checks
    # ------------------------------------------------------------------

    def _check_constraints(
        self, model: Any, data: Any, step: int
    ) -> list[dict[str, Any]]:
        """Check all safety constraints on the given state."""
        violations: list[dict[str, Any]] = []

        # Joint limits
        for jid in range(model.njnt):
            qpos_addr = model.jnt_qposadr[jid]
            qvel_addr = model.jnt_dofadr[jid]
            joint_type = model.jnt_type[jid]
            import mujoco
            if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                pos = float(data.qpos[qpos_addr])
                limited = model.jnt_limited[jid]
                if limited:
                    lo = float(model.jnt_range[jid][0])
                    hi = float(model.jnt_range[jid][1])
                    if pos < lo - 0.01 or pos > hi + 0.01:
                        name_start = model.name_jntadr[jid]
                        name = model.names[name_start:].decode("utf-8").split("\x00")[0]
                        violations.append({
                            "step": step,
                            "type": "joint_limit",
                            "joint": name or f"joint_{jid}",
                            "position": pos,
                            "limit": [lo, hi],
                        })

        # ZMP — approximate from contact forces
        zmp_x, zmp_y = self._compute_zmp_from_data(model, data)
        if not self._point_in_polygon((zmp_x, zmp_y), self._zmp_polygon):
            violations.append({
                "step": step,
                "type": "zmp_violation",
                "zmp": [zmp_x, zmp_y],
            })

        # Collision force
        total_force = 0.0
        for ci in range(data.ncon):
            force = np.zeros(6)
            import mujoco
            mujoco.mj_contactForce(model, data, ci, force)
            mag = float(np.linalg.norm(force[:3]))
            total_force += mag
        if total_force > self._collision_force_threshold:
            violations.append({
                "step": step,
                "type": "collision_force",
                "force": total_force,
                "threshold": self._collision_force_threshold,
            })

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_frame(self, model: Any, data: Any, step: int) -> dict[str, Any]:
        """Build a minimal snapshot dict for a rollout frame."""
        return {
            "step": step,
            "time": float(data.time),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "com_position": data.subtree_com[0].copy().tolist() if data.subtree_com.shape[0] > 0 else [0.0, 0.0, 0.0],
        }

    @staticmethod
    def _compute_zmp_from_data(model: Any, data: Any) -> tuple[float, float]:
        """Compute ZMP from MuJoCo contact forces."""
        import mujoco
        total_fz = 0.0
        zx_sum = 0.0
        zy_sum = 0.0
        for i in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            fz = force[2]
            if fz > 1e-6:
                pos = data.contact[i].pos
                total_fz += fz
                zx_sum += fz * pos[0]
                zy_sum += fz * pos[1]
        if total_fz > 1e-6:
            return (zx_sum / total_fz, zy_sum / total_fz)
        return (0.0, 0.0)

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        """Ray-casting point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _compute_uncertainty(
        self,
        violations: list[dict[str, Any]],
        steps: int,
        action_sequence: list[np.ndarray],
    ) -> float:
        """Heuristic uncertainty score.

        Increases with:
        - Number of safety violations
        - Large control magnitudes (aggressive actions)
        - Empty action sequences
        """
        if steps == 0:
            return 1.0

        base = 0.1  # minimum uncertainty
        violation_penalty = min(0.5, len(violations) * 0.05)

        # Control magnitude penalty
        ctrl_mag = 0.0
        if action_sequence:
            mags = [float(np.linalg.norm(c)) for c in action_sequence[:steps]]
            ctrl_mag = sum(mags) / max(len(mags), 1)
        ctrl_penalty = min(0.4, ctrl_mag * 0.1)

        return min(1.0, base + violation_penalty + ctrl_penalty)
