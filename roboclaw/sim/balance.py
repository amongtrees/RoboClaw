"""Shared balance controller — torque control + elastic band + gravity compensation.

This module extracts the working standing-balance loop from
``scripts/view_mujoco.py`` into a reusable ``BalanceController`` class that
every MuJoCo executor calls at each physics step.

For H1 (gaintype=0, direct torque control), the control law is::

    ctrl = qfrc_bias[6:] + kp*(q_des - q) - kd*qd + ankle_com_fb

plus an elastic band on the pelvis that provides ~65 % weight support.

Controllers (WalkingController, GraspingController, …) are responsible for
generating *q_des* trajectories and *joint_mask* arrays.  This module
handles the dynamics — torque computation, balance, and ground reaction
force injection.

Joint index reference (H1, 19 actuators)
----------------------------------------
.. list-table::
   :header-rows: 1

   * - Indices
     - Group
     - Names
   * - 0-4
     - Left leg
     - hip_yaw, hip_roll, hip_pitch, knee, ankle
   * - 5-9
     - Right leg
     - hip_yaw, hip_roll, hip_pitch, knee, ankle
   * - 10
     - Torso
     - torso
   * - 11-14
     - Left arm
     - shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
   * - 15-18
     - Right arm
     - shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint mask helpers — dynamically sized for the actual actuator count
# ---------------------------------------------------------------------------


def make_joint_masks(n_actuators: int) -> dict[str, np.ndarray]:
    """Create joint masks for a robot with *n_actuators* actuators.

    Assumes H1-style ordering: legs[0..9], torso[10], arms[11..17]
    when n_actuators >= 19.  For smaller models, masks are built
    heuristically from the first/last indices.
    """
    n = n_actuators
    n_leg = min(n, 10)  # up to 10 leg joints
    n_arm = max(0, n - 11)  # remaining are arms
    n_arm_half = max(0, n_arm // 2)

    masks: dict[str, np.ndarray] = {
        "none": np.zeros(n, dtype=np.float64),
    }

    # Legs + torso
    leg_torso = np.zeros(n, dtype=np.float64)
    leg_torso[:n_leg] = 1.0
    if n > n_leg:
        leg_torso[n_leg] = 1.0  # torso
    masks["legs_torso"] = leg_torso

    # Legs only
    legs = np.zeros(n, dtype=np.float64)
    legs[:n_leg] = 1.0
    masks["legs"] = legs

    # Torso only
    if n > n_leg:
        torso = np.zeros(n, dtype=np.float64)
        torso[n_leg] = 1.0
        masks["torso"] = torso

    # Arms
    arms = np.zeros(n, dtype=np.float64)
    arms[n_leg + 1:] = 1.0
    masks["arms"] = arms

    # Right arm (second half)
    right = np.zeros(n, dtype=np.float64)
    right[n_leg + 1 + n_arm_half:n_leg + 1 + n_arm] = 1.0
    masks["right_arm"] = right

    # Left arm (first half)
    left = np.zeros(n, dtype=np.float64)
    left[n_leg + 1:n_leg + 1 + n_arm_half] = 1.0
    masks["left_arm"] = left

    # Ankles
    ankles = np.zeros(n, dtype=np.float64)
    if n_leg >= 5:
        ankles[4] = 1.0  # left ankle
    if n_leg >= 10:
        ankles[9] = 1.0  # right ankle
    masks["ankles"] = ankles

    return masks


# Module-level constants for H1 (19 actuators) — convenience
_H1_MASKS = make_joint_masks(19)
MASK_LEGS_TORSO = _H1_MASKS["legs_torso"]
MASK_LEGS_ONLY = _H1_MASKS["legs"]
MASK_TORSO = _H1_MASKS["torso"]
MASK_ARMS = _H1_MASKS["arms"]
MASK_RIGHT_ARM = _H1_MASKS["right_arm"]
MASK_LEFT_ARM = _H1_MASKS["left_arm"]
MASK_NONE = _H1_MASKS["none"]
MASK_ANKLES = _H1_MASKS["ankles"]

# Leg joint indices (0-9) + torso (10) — for H1; dynamic in BalanceController
LEG_INDICES = list(range(11))
ARM_INDICES = list(range(11, 19))

# Ankle indices in the actuator array — H1 defaults
ANKLE_LEFT_IDX = 4
ANKLE_RIGHT_IDX = 9


# ---------------------------------------------------------------------------
# BalanceController
# ---------------------------------------------------------------------------


class BalanceController:
    """Shared torque-control backbone for all H1 MuJoCo skills.

    Provides per-step torque computation with:
    - Elastic band on pelvis (~65 % weight support + height feedback)
    - Gravity compensation via ``qfrc_bias[6:]``
    - Joint-space PD control: ``kp*(q_des - q) - kd*qd``
    - Ankle COM feedback for sagittal balance
    - Horizontal centering (pelvis PD in x/y)

    Parameters
    ----------
    env:
        The MuJoCo environment (must be loaded with the H1 scene XML).
        If ``None``, ``model`` and ``data`` kwargs must be provided.
    model:
        MuJoCo model (used when env is None).
    data:
        MuJoCo data (used when env is None).
    kp:
        Position gain for joint PD (default 150).
    kd:
        Velocity damping gain (default 15).
    band_ratio:
        Fraction of total weight supported by the elastic band (default 0.65).
    band_kp_xy:
        Horizontal centering stiffness (default 500 N/m).
    band_kd_xy:
        Horizontal centering damping (default 100 N·s/m).
    com_fb_gain:
        Ankle COM feedback gain (default 60 Nm/m).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        env: Any = None,
        *,
        model: Any = None,
        data: Any = None,
        kp: float = 150.0,
        kd: float = 15.0,
        band_ratio: float = 0.65,
        band_kp_xy: float = 500.0,
        band_kd_xy: float = 100.0,
        com_fb_gain: float = 60.0,
    ) -> None:
        import mujoco

        # Support both env-based and direct model/data construction
        if env is not None:
            self._model = env.model
            self._data = env.data
        elif model is not None and data is not None:
            self._model = model
            self._data = data
        else:
            raise ValueError("BalanceController requires either 'env' or both 'model' and 'data'")
        self._kp = kp
        self._kd = kd
        self._band_ratio = band_ratio
        self._band_kp_xy = band_kp_xy
        self._band_kd_xy = band_kd_xy
        self._com_fb_gain = com_fb_gain

        # --- Load standing pose from "home" keyframe ---
        home_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, home_id)
        else:
            # No keyframe — forward kinematics to get initial state
            mujoco.mj_forward(self._model, self._data)

        # Lower pelvis so feet touch ground (H1-specific)
        self._lower_to_ground()

        # Record standing qpos (actuator space: qpos[7:] for free-joint models)
        free_joint_qpos_size = 7 if self._model.nq > self._model.nu else 0
        self.standing_qpos = self._data.qpos[free_joint_qpos_size:].copy()
        if len(self.standing_qpos) != self._model.nu:
            # Model may not have a free joint — use all qpos as standing
            self.standing_qpos = self._data.qpos[:self._model.nu].copy()

        # Find pelvis or root body
        try:
            self._pelvis_id = self._model.body("pelvis").id
        except KeyError:
            # Fall back to the first non-world body (usually torso)
            self._pelvis_id = 1  # world is 0, first body is 1
        self._pelvis_target_z = float(self._data.qpos[2])

        # Pre-compute reference gravity torques at the adjusted standing pose
        mujoco.mj_forward(self._model, self._data)
        bias_offset = free_joint_qpos_size if free_joint_qpos_size else 0
        g_ref = self._data.qfrc_bias[bias_offset:].copy()
        self._g_comp_ref = g_ref if len(g_ref) == self._model.nu else np.zeros(self._model.nu)

        # Robot properties
        self._total_mass = self._compute_total_mass()
        self._weight = self._total_mass * 9.81
        self._band_fz = self._band_ratio * self._weight

        # Identify ankle indices from actuator names (dynamic — works for any model)
        self._ankle_left_idx = self._find_actuator_index(
            ["left_ankle", "left_ankle_act", "ankle_left"]
        )
        self._ankle_right_idx = self._find_actuator_index(
            ["right_ankle", "right_ankle_act", "ankle_right"]
        )

    def _find_actuator_index(self, candidates: list[str]) -> int | None:
        """Find the actuator index matching any of the candidate names."""
        for c in candidates:
            for i in range(self._model.nu):
                s = self._model.name_actuatoradr[i]
                name = self._model.names[s:].decode("utf-8").split("\x00")[0]
                if c in name or name in c:
                    return i
        return None

        logger.info(
            "BalanceController ready: mass=%.1f kg, band=%.0f N (%.0f%% weight), "
            "pelvis_z=%.3f",
            self._total_mass,
            self._band_fz,
            self._band_ratio * 100,
            self._pelvis_target_z,
        )

    # ------------------------------------------------------------------
    # Per-step control
    # ------------------------------------------------------------------

    def step_control(
        self,
        env: Any = None,
        *,
        model: Any = None,
        data: Any = None,
        skill_q_des: np.ndarray | None = None,
        joint_mask: np.ndarray | None = None,
        external_force: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Compute the torque control vector for one physics step.

        Parameters
        ----------
        env:
            The MuJoCo environment.  When None, uses ``self._model`` and
            ``self._data`` (set at construction time).
        model, data:
            Direct model/data access (overrides env, used for viewer sharing).
        skill_q_des:
            Absolute joint position targets from a skill controller,
            shape ``(19,)`` in actuator order.  Non-controlled joints
            should be 0.0 (they will be replaced by the standing q_des
            via the joint_mask blend).
        joint_mask:
            Float array of shape ``(19,)`` where ``1.0`` means "use the
            skill q_des for this joint" and ``0.0`` means "use the
            standing q_des".  Pass ``MASK_NONE`` (all zeros) to stand
            still.  When ``None``, all joints track the standing pose.
        external_force:
            Optional (fx, fy, fz) force applied to the pelvis IN ADDITION
            to the elastic band.  Used for locomotion propulsion.

        Returns
        -------
        ctrl : np.ndarray
            Torque control array of shape ``(19,)`` ready for
            ``data.ctrl[:]``.
        """
        import mujoco

        _model = model or (env.model if env else self._model)
        _data = data or (env.data if env else self._data)

        # 1. Blend standing q_des with skill q_des -------------------------
        if skill_q_des is not None and joint_mask is not None:
            q_des = joint_mask * skill_q_des + (1.0 - joint_mask) * self.standing_qpos
        elif skill_q_des is not None:
            q_des = np.asarray(skill_q_des, dtype=np.float64)
        else:
            q_des = self.standing_qpos.copy()

        # 2. Elastic band on pelvis ----------------------------------------
        pelvis_pos = _data.xpos[self._pelvis_id]
        pelvis_vel = _data.cvel[self._pelvis_id][3:6]  # linear velocity

        # Vertical: weight support + height feedback
        dz = self._pelvis_target_z - pelvis_pos[2]
        fz = self._band_fz + 800.0 * dz - 150.0 * pelvis_vel[2]
        fz = max(0.0, fz)  # band pulls up only

        # Horizontal: PD centering
        fx = -self._band_kp_xy * pelvis_pos[0] - self._band_kd_xy * pelvis_vel[0]
        fy = -self._band_kp_xy * pelvis_pos[1] - self._band_kd_xy * pelvis_vel[1]

        _data.xfrc_applied[self._pelvis_id, :3] = [fx, fy, fz]
        _data.xfrc_applied[self._pelvis_id, 3:] = 0.0

        # 2b. External force (e.g. forward push for locomotion) ----------
        if external_force is not None:
            _data.xfrc_applied[self._pelvis_id, 0] += float(external_force[0])
            _data.xfrc_applied[self._pelvis_id, 1] += float(external_force[1])
            _data.xfrc_applied[self._pelvis_id, 2] += float(external_force[2])

        # 3. Torque control: PD + live gravity compensation -----------------
        # For free-joint models (H1), actuator qpos starts at index 7.
        # For fixed-base models (test humanoid), it starts at 0.
        q_offset = _data.qpos.shape[0] - self._model.nu if _data.qpos.shape[0] > self._model.nu else 0
        q = _data.qpos[q_offset:q_offset + self._model.nu]
        qd = _data.qvel[q_offset - 1:q_offset - 1 + self._model.nu] if q_offset > 0 else _data.qvel[:self._model.nu]
        if len(qd) != self._model.nu:
            qd = np.zeros(self._model.nu)  # fallback
        g_comp = _data.qfrc_bias[q_offset:q_offset + self._model.nu].copy()
        if len(g_comp) != self._model.nu:
            g_comp = np.zeros(self._model.nu)

        ctrl = g_comp + self._kp * (q_des - q) - self._kd * qd

        # 4. Ankle COM feedback for sagittal balance -----------------------
        com = _data.subtree_com[0] if _data.subtree_com.shape[0] > 0 else np.zeros(3)
        com_err_x = com[0]
        if self._ankle_left_idx is not None and self._ankle_left_idx < len(ctrl):
            ctrl[self._ankle_left_idx] -= self._com_fb_gain * com_err_x
        if self._ankle_right_idx is not None and self._ankle_right_idx < len(ctrl):
            ctrl[self._ankle_right_idx] -= self._com_fb_gain * com_err_x

        # 5. Clip to actuator torque limits --------------------------------
        ctrl = self._clip_torques(ctrl)

        return ctrl

    # ------------------------------------------------------------------
    # Settling
    # ------------------------------------------------------------------

    def settle(
        self,
        env: Any = None,
        *,
        model: Any = None,
        data: Any = None,
        steps: int = 300,
        verbose: bool = False,
    ) -> None:
        """Run standing balance for *steps* physics steps.

        Call once after environment creation and after every reset.

        Parameters
        ----------
        env:
            MuJoCo environment (or None to use stored model/data).
        model, data:
            Direct model/data (overrides env).
        steps:
            Number of physics steps to run.
        verbose:
            If True, log final pelvis/COM position.
        """
        import mujoco

        _model = model or (env.model if env else self._model)
        _data = data or (env.data if env else self._data)

        for i in range(steps):
            ctrl = self.step_control(model=_model, data=_data)
            _data.ctrl[:] = ctrl
            mujoco.mj_step(_model, _data)

        if verbose:
            com = _data.subtree_com[0] if _data.subtree_com.shape[0] > 0 else [0, 0, 0]
            pelvis_z = _data.xpos[self._pelvis_id][2]
            logger.info(
                "BalanceController settled (%d steps): pelvis_z=%.3f, COM=[%.3f,%.3f,%.3f]",
                steps, pelvis_z, com[0], com[1], com[2],
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lower_to_ground(self) -> None:
        """Adjust pelvis Z so the lowest foot sole just touches z=0."""
        import mujoco

        mujoco.mj_forward(self._model, self._data)
        lowest = 999.0
        for foot_name in ("left_ankle_link", "right_ankle_link"):
            try:
                bid = self._model.body(foot_name).id
                # Foot collision geom extends ~0.058 m below ankle origin
                sole_z = self._data.xpos[bid][2] - 0.058
                if sole_z < lowest:
                    lowest = sole_z
            except KeyError:
                continue
        if lowest < 999.0:
            self._data.qpos[2] -= lowest
            mujoco.mj_forward(self._model, self._data)
            logger.debug("Pelvis lowered by %.4f m to touch ground", lowest)

    def _compute_total_mass(self) -> float:
        """Sum all body masses excluding world (index 0)."""
        total = 0.0
        for bid in range(1, self._model.nbody):
            total += self._model.body_mass[bid]
        return total

    def _clip_torques(self, ctrl: np.ndarray) -> np.ndarray:
        """Clip torques to actuator limits defined in the model."""
        # MuJoCo stores ctrlrange as [min, max] for each actuator
        for i in range(min(len(ctrl), self._model.nu)):
            lo = self._model.actuator_ctrlrange[i][0]
            hi = self._model.actuator_ctrlrange[i][1]
            if lo != 0.0 or hi != 0.0:  # (0,0) means unbounded
                ctrl[i] = max(lo, min(hi, ctrl[i]))
        return ctrl

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def standing_q_des(self) -> np.ndarray:
        """Return a copy of the standing pose q_des (shape 19)."""
        return self.standing_qpos.copy()

    @property
    def pelvis_id(self) -> int:
        return self._pelvis_id

    @property
    def pelvis_target_z(self) -> float:
        return self._pelvis_target_z

    @property
    def total_mass(self) -> float:
        return self._total_mass

    @property
    def band_force_z(self) -> float:
        """Nominal vertical band force (N)."""
        return self._band_fz

    @property
    def kp(self) -> float:
        return self._kp

    @property
    def kd(self) -> float:
        return self._kd
