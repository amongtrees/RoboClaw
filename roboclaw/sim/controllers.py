"""Robot skill controllers — generate q_des trajectories for MuJoCo execution.

Each controller encapsulates the *kinematic* plan for one or more skills.
They produce **absolute joint position targets** (q_des, shape 19 for H1)
and a **joint_mask** that tells the ``BalanceController`` which joints
are actively controlled by this skill.  Non-controlled joints stay at
the standing pose (via the mask blend in ``BalanceController.step_control``).

All ``plan_*`` methods return ``(trajectory, joint_mask)`` where:

* **trajectory**: ``list[np.ndarray]`` — each element is a 19-element
  absolute q_des array in actuator order.  Joints *not* controlled by
  this skill should be ``0.0`` (the ``BalanceController`` blends them
  with the standing q_des via the joint mask).
* **joint_mask**: ``np.ndarray`` of shape (19,) with ``1.0`` for
  controlled joints and ``0.0`` for joints that should track the
  standing pose.

Torque computation (gravity compensation, PD control, elastic band,
COM feedback) is handled entirely by ``BalanceController`` — controllers
in this module are purely kinematic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from roboclaw.sim.balance import make_joint_masks

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _interpolate_linear(
    start: np.ndarray,
    end: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    """Linear interpolation between two joint-space targets."""
    if num_steps <= 1:
        return [end.copy()]
    trajectory = []
    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        trajectory.append(start + alpha * (end - start))
    return trajectory


# ---------------------------------------------------------------------------
# WalkingController
# ---------------------------------------------------------------------------


class WalkingController:
    """CPG-based bipedal walking controller for full H1 leg DOF.

    Generates sinusoidal trajectories for hip pitch (swing), knee (lift),
    and ankle (push-off), plus hip roll for lateral balance and hip yaw
    for pelvic rotation.  The torso is given a small counter-rotation.

    Parameters
    ----------
    joint_names:
        Ordered list of actuator names (length 19 for H1).
    standing_qpos:
        Standing pose q_des (19,).  Used as the baseline for the CPG
        trajectory — the sine waves are **absolute** joint targets
        centered on the standing pose values.
    step_height:
        Nominal step height in meters (affects knee amplitude).
    max_speed:
        Maximum walking speed in m/s.
    """

    # Joint names (H1 actuator names, used for index lookup)
    HIP_YAW_L = "left_hip_yaw"
    HIP_ROLL_L = "left_hip_roll"
    HIP_PITCH_L = "left_hip_pitch"
    KNEE_L = "left_knee"
    ANKLE_L = "left_ankle"
    HIP_YAW_R = "right_hip_yaw"
    HIP_ROLL_R = "right_hip_roll"
    HIP_PITCH_R = "right_hip_pitch"
    KNEE_R = "right_knee"
    ANKLE_R = "right_ankle"
    TORSO = "torso"

    def __init__(
        self,
        joint_names: list[str],
        standing_qpos: np.ndarray | None = None,
        step_height: float = 0.05,
        max_speed: float = 1.5,
    ) -> None:
        self._joint_names = list(joint_names)
        self._step_height = step_height
        self._max_speed = max_speed
        self._n_joints = len(joint_names)

        # Standing pose baseline for CPG center
        if standing_qpos is not None:
            self._standing_qpos = np.asarray(standing_qpos, dtype=np.float64)
        else:
            self._standing_qpos = np.zeros(self._n_joints, dtype=np.float64)

        # Build index map (strip _act suffix if present)
        self._idx: dict[str, int] = {}
        for i, name in enumerate(self._joint_names):
            jname = name.replace("_act", "")
            self._idx[jname] = i

        # Build dynamic masks for this robot
        _masks = make_joint_masks(self._n_joints)
        self._mask_legs_torso = _masks.get("legs_torso", _masks.get("legs", _masks["none"]))

    # ------------------------------------------------------------------
    # Public API — all return (trajectory, joint_mask)
    # ------------------------------------------------------------------

    def plan_steps(
        self,
        num_steps: int,
        direction: str = "forward",
        step_length: float = 0.3,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a joint-space trajectory for walking *num_steps* steps.

        The robot walks by leaning forward (hip pitch bias) and alternating
        leg lift.  The forward lean shifts the COM ahead of the feet,
        creating a natural falling-forward that the stepping legs catch.
        Ground reaction forces then drive forward progression.

        Returns:
            (trajectory, self._mask_legs_torso)
        """
        freq = 1.2  # Hz — slightly faster gait for better dynamics
        ctrl_rate = 50.0  # assumed control rate (Hz); matches 0.02s step
        period_steps = max(1, int(ctrl_rate / freq))
        total_ctrl_steps = num_steps * period_steps

        # Amplitudes — scaled by step_length relative to nominal 0.3 m
        scale = min(step_length / 0.3, 1.5)
        # Forward lean: shifts COM ahead of feet, creates gravity moment
        forward_lean = 0.15 * scale
        hip_pitch_amp = 0.25 * scale  # swing amplitude
        knee_amp = 0.55 * scale       # foot clearance
        ankle_amp = 0.12 * scale      # push-off at toe-off
        hip_roll_amp = 0.05           # lateral weight shift
        hip_yaw_amp = 0.02            # pelvic rotation

        # Forward propulsion force applied to pelvis each step.
        # Because we use position control, the sinusoidal leg motion
        # alone doesn't generate net forward force — we need an
        # explicit push to overcome friction and centering forces.
        push_force_per_step = 35.0 * scale  # N per leg per step

        if direction == "backward":
            forward_lean *= -1.0
            hip_pitch_amp *= -1.0
        elif direction == "left":
            forward_lean = 0.0
            hip_pitch_amp *= 0.2
            hip_roll_amp *= 2.0  # stronger lateral sway
        elif direction == "right":
            forward_lean = 0.0
            hip_pitch_amp *= 0.2
            hip_roll_amp *= -2.0  # opposite lateral sway

        trajectory: list[np.ndarray] = []
        for t in range(total_ctrl_steps):
            phase = 2.0 * math.pi * t / period_steps
            ctrl = self._standing_qpos.copy()

            # --- Left leg (phase = 0 at start of left swing) ---
            l_hip = self._idx.get(self.HIP_PITCH_L, 0)
            l_knee = self._idx.get(self.KNEE_L, 0)
            l_ankle = self._idx.get(self.ANKLE_L, 0)

            # Hip pitch: swing oscillation + forward lean (propulsion)
            self._set(ctrl, self.HIP_PITCH_L,
                      self._standing_qpos[l_hip]
                      + forward_lean
                      + hip_pitch_amp * math.sin(phase))
            # Knee: lift during swing (positive = flex), near-zero in stance
            self._set(ctrl, self.KNEE_L,
                      self._standing_qpos[l_knee]
                      + knee_amp * max(0.0, math.sin(phase)))
            # Ankle: dorsiflex during swing (toe clearance),
            #        plantarflex during stance (push-off)
            self._set(ctrl, self.ANKLE_L,
                      self._standing_qpos[l_ankle]
                      - ankle_amp * math.sin(phase + math.pi * 0.3))
            # Hip roll: lateral weight shift
            self._set(ctrl, self.HIP_ROLL_L,
                      self._standing_qpos[self._idx.get(self.HIP_ROLL_L, 0)]
                      + hip_roll_amp * math.cos(phase))
            # Hip yaw: slight pelvic rotation
            self._set(ctrl, self.HIP_YAW_L,
                      self._standing_qpos[self._idx.get(self.HIP_YAW_L, 0)]
                      + hip_yaw_amp * math.sin(phase * 0.5))

            # --- Right leg (phase = π, anti-phase to left) ---
            r_hip = self._idx.get(self.HIP_PITCH_R, 0)
            r_knee = self._idx.get(self.KNEE_R, 0)
            r_ankle = self._idx.get(self.ANKLE_R, 0)

            self._set(ctrl, self.HIP_PITCH_R,
                      self._standing_qpos[r_hip]
                      + forward_lean
                      + hip_pitch_amp * math.sin(phase + math.pi))
            self._set(ctrl, self.KNEE_R,
                      self._standing_qpos[r_knee]
                      + knee_amp * max(0.0, math.sin(phase + math.pi)))
            self._set(ctrl, self.ANKLE_R,
                      self._standing_qpos[r_ankle]
                      - ankle_amp * math.sin(phase + math.pi + math.pi * 0.3))
            self._set(ctrl, self.HIP_ROLL_R,
                      self._standing_qpos[self._idx.get(self.HIP_ROLL_R, 0)]
                      + hip_roll_amp * math.cos(phase + math.pi))
            self._set(ctrl, self.HIP_YAW_R,
                      self._standing_qpos[self._idx.get(self.HIP_YAW_R, 0)]
                      + hip_yaw_amp * math.sin(phase * 0.5 + math.pi))

            # Torso: slight counter-rotation for balance
            self._set(ctrl, self.TORSO,
                      self._standing_qpos[self._idx.get(self.TORSO, 0)]
                      - 0.02 * math.sin(phase))

            trajectory.append(ctrl)

        return trajectory, self._mask_legs_torso

    def plan_navigate(self, target_distance: float) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a walking trajectory to cover *target_distance* meters.

        Returns:
            (trajectory, self._mask_legs_torso)
        """
        step_length = min(0.3, self._max_speed * 0.5)
        num_steps = max(1, int(target_distance / step_length))
        return self.plan_steps(num_steps, "forward", step_length)

    def plan_climb(
        self, num_stairs: int, direction: str = "up"
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a stair-climbing trajectory.

        Returns:
            (trajectory, self._mask_legs_torso)
        """
        freq = 0.6
        period_steps = max(1, int(1.0 / freq / 0.05))
        total_ctrl_steps = num_stairs * period_steps

        # Higher amplitudes for stair climbing
        hip_pitch_amp = 0.5
        knee_amp = 1.0
        ankle_amp = 0.25
        hip_roll_amp = 0.06
        hip_yaw_amp = 0.02

        if direction == "down":
            hip_pitch_amp *= -1.0
            knee_amp *= 0.6

        trajectory: list[np.ndarray] = []
        for t in range(total_ctrl_steps):
            phase = 2.0 * math.pi * t / period_steps
            ctrl = self._standing_qpos.copy()

            self._set(ctrl, self.HIP_PITCH_L,
                      self._standing_qpos[self._idx.get(self.HIP_PITCH_L, 0)]
                      + hip_pitch_amp * math.sin(phase))
            self._set(ctrl, self.KNEE_L,
                      self._standing_qpos[self._idx.get(self.KNEE_L, 0)]
                      + knee_amp * max(0.0, math.sin(phase)))
            self._set(ctrl, self.ANKLE_L,
                      self._standing_qpos[self._idx.get(self.ANKLE_L, 0)]
                      + ankle_amp * math.sin(phase + math.pi * 0.5))
            self._set(ctrl, self.HIP_ROLL_L,
                      self._standing_qpos[self._idx.get(self.HIP_ROLL_L, 0)]
                      + hip_roll_amp * math.sin(phase + math.pi * 0.5))
            self._set(ctrl, self.HIP_YAW_L,
                      self._standing_qpos[self._idx.get(self.HIP_YAW_L, 0)]
                      + hip_yaw_amp * math.sin(phase * 0.5))

            self._set(ctrl, self.HIP_PITCH_R,
                      self._standing_qpos[self._idx.get(self.HIP_PITCH_R, 0)]
                      + hip_pitch_amp * math.sin(phase + math.pi))
            self._set(ctrl, self.KNEE_R,
                      self._standing_qpos[self._idx.get(self.KNEE_R, 0)]
                      + knee_amp * max(0.0, math.sin(phase + math.pi)))
            self._set(ctrl, self.ANKLE_R,
                      self._standing_qpos[self._idx.get(self.ANKLE_R, 0)]
                      + ankle_amp * math.sin(phase + math.pi + math.pi * 0.5))
            self._set(ctrl, self.HIP_ROLL_R,
                      self._standing_qpos[self._idx.get(self.HIP_ROLL_R, 0)]
                      + hip_roll_amp * math.sin(phase + math.pi * 0.5))
            self._set(ctrl, self.HIP_YAW_R,
                      self._standing_qpos[self._idx.get(self.HIP_YAW_R, 0)]
                      + hip_yaw_amp * math.sin(phase * 0.5 + math.pi))

            self._set(ctrl, self.TORSO,
                      self._standing_qpos[self._idx.get(self.TORSO, 0)]
                      - 0.03 * math.sin(phase))

            trajectory.append(ctrl)

        return trajectory, self._mask_legs_torso

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set(self, ctrl: np.ndarray, joint_name: str, value: float) -> None:
        """Set a control value for a named joint if it exists."""
        idx = self._idx.get(joint_name)
        if idx is not None:
            ctrl[idx] = float(value)

    @property
    def joint_mask(self) -> np.ndarray:
        """The mask for locomotion skills (legs + torso)."""
        return self._mask_legs_torso.copy()


# ---------------------------------------------------------------------------
# GraspingController
# ---------------------------------------------------------------------------


class GraspingController:
    """Arm controller with full shoulder + elbow DOF for H1-style robots.

    Generates reaching, grasping, placing, and door-opening trajectories
    controlling all 8 arm joints.  Uses simple inverse kinematics
    heuristics based on shoulder-relative target vectors.

    Parameters
    ----------
    joint_names:
        Ordered list of actuator names (length 19 for H1).
    standing_qpos:
        Standing pose q_des (19,).  Used as the baseline for arm
        trajectory targets.
    arm_workspace_radius:
        Nominal arm reach in meters.
    """

    # H1 arm joints (actuator names)
    SHOULDER_PITCH_L = "left_shoulder_pitch"
    SHOULDER_ROLL_L = "left_shoulder_roll"
    SHOULDER_YAW_L = "left_shoulder_yaw"
    ELBOW_L = "left_elbow"
    SHOULDER_PITCH_R = "right_shoulder_pitch"
    SHOULDER_ROLL_R = "right_shoulder_roll"
    SHOULDER_YAW_R = "right_shoulder_yaw"
    ELBOW_R = "right_elbow"

    def __init__(
        self,
        joint_names: list[str],
        standing_qpos: np.ndarray | None = None,
        arm_workspace_radius: float = 0.85,
    ) -> None:
        self._joint_names = list(joint_names)
        self._n_joints = len(joint_names)
        self._workspace = arm_workspace_radius

        if standing_qpos is not None:
            self._standing_qpos = np.asarray(standing_qpos, dtype=np.float64)
        else:
            self._standing_qpos = np.zeros(self._n_joints, dtype=np.float64)

        # Build index map
        self._idx: dict[str, int] = {}
        for i, name in enumerate(self._joint_names):
            jname = name.replace("_act", "")
            self._idx[jname] = i

        # Build dynamic masks for this robot
        _masks = make_joint_masks(self._n_joints)
        self._mask_right = _masks.get("right_arm", _masks["none"])
        self._mask_left = _masks.get("left_arm", _masks["none"])
        self._mask_arms = _masks.get("arms", _masks["none"])

    # ------------------------------------------------------------------
    # Public API — all return (trajectory, joint_mask)
    # ------------------------------------------------------------------

    def plan_reach(
        self,
        target_xyz: tuple[float, float, float],
        arm: str = "right",
        num_steps: int = 50,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a trajectory to reach toward a target position.

        Returns:
            (trajectory, joint_mask) — mask is right_arm or left_arm.
        """
        tx, ty, tz = target_xyz

        # Shoulder base position (approximate, relative to torso)
        if arm == "left":
            shoulder_y = -0.25
            shoulder_z = 1.45
            dy = ty - shoulder_y
            dz = tz - shoulder_z
            dx = tx - 0.0
            shoulder_pitch = math.atan2(-dx, dz)
            shoulder_roll = math.atan2(dy, abs(dz)) * 0.5
            shoulder_yaw = 0.0
            # Elbow flexion — more bend when reaching close
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            elbow = -1.2 - 0.3 * min(dist / self._workspace, 1.0)
            mask = self._mask_left
        else:
            shoulder_y = 0.25
            shoulder_z = 1.45
            dy = ty - shoulder_y
            dz = tz - shoulder_z
            dx = tx - 0.0
            shoulder_pitch = math.atan2(-dx, dz)
            shoulder_roll = math.atan2(dy, abs(dz)) * 0.5
            shoulder_yaw = 0.0
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            elbow = -1.2 - 0.3 * min(dist / self._workspace, 1.0)
            mask = self._mask_right

        # Clamp to reasonable joint limits
        shoulder_pitch = max(-2.0, min(2.0, shoulder_pitch))
        shoulder_roll = max(-1.5, min(1.5, shoulder_roll))
        shoulder_yaw = max(-1.0, min(1.0, shoulder_yaw))
        elbow = max(-2.5, min(0.0, elbow))

        # Build target vector (absolute, starting from standing pose)
        target_ctrl = self._standing_qpos.copy()
        if arm == "left":
            self._set(target_ctrl, self.SHOULDER_PITCH_L, shoulder_pitch)
            self._set(target_ctrl, self.SHOULDER_ROLL_L, shoulder_roll)
            self._set(target_ctrl, self.SHOULDER_YAW_L, shoulder_yaw)
            self._set(target_ctrl, self.ELBOW_L, elbow)
        else:
            self._set(target_ctrl, self.SHOULDER_PITCH_R, shoulder_pitch)
            self._set(target_ctrl, self.SHOULDER_ROLL_R, shoulder_roll)
            self._set(target_ctrl, self.SHOULDER_YAW_R, shoulder_yaw)
            self._set(target_ctrl, self.ELBOW_R, elbow)

        # Interpolate from standing to target
        start_vec = self._standing_qpos.copy()
        trajectory = _interpolate_linear(start_vec, target_ctrl, num_steps)

        return trajectory, mask

    def plan_grasp(
        self,
        object_pos: tuple[float, float, float],
        arm: str = "right",
        num_steps: int = 50,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a grasp trajectory: approach from 5 cm above, then close in.

        Returns:
            (trajectory, joint_mask)
        """
        ox, oy, oz = object_pos
        # Phase 1: approach from above (40% of steps)
        approach = (ox, oy, oz + 0.08)
        traj1, mask = self.plan_reach(approach, arm, num_steps=max(1, int(num_steps * 0.4)))
        # Phase 2: descend to object (60% of steps)
        traj2, _ = self.plan_reach(object_pos, arm, num_steps=max(1, int(num_steps * 0.6)))

        # If traj1 ends at the approach target and traj2 starts there,
        # we can concatenate.  Otherwise just use the descent phase.
        combined = traj1 + traj2
        return combined, mask

    def plan_place(
        self,
        target_xyz: tuple[float, float, float],
        arm: str = "right",
        num_steps: int = 40,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a placing trajectory.

        Returns:
            (trajectory, joint_mask)
        """
        return self.plan_reach(target_xyz, arm, num_steps=num_steps)

    def plan_door_open(
        self,
        door_handle_pos: tuple[float, float, float] | None = None,
        arm: str = "right",
        num_steps: int = 80,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a door-opening trajectory: reach handle, then pull back.

        Returns:
            (trajectory, joint_mask)
        """
        if door_handle_pos is None:
            door_handle_pos = (0.5, 0.25, 1.0)

        hx, hy, hz = door_handle_pos

        # Phase 1: reach to handle (50% of steps)
        traj1, mask = self.plan_reach(
            door_handle_pos, arm, num_steps=max(1, int(num_steps * 0.5))
        )

        # Phase 2: pull handle toward robot (simulating door swing)
        # Pull back in x and y (door opens toward robot)
        pull_target = (hx - 0.3, hy - 0.15 if arm == "right" else hy + 0.15, hz)
        traj2, _ = self.plan_reach(
            pull_target, arm, num_steps=max(1, int(num_steps * 0.5))
        )

        return traj1 + traj2, mask

    def plan_handover(
        self,
        receiver_pos: tuple[float, float, float] | None = None,
        arm: str = "right",
        num_steps: int = 60,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a handover trajectory: extend arm forward at chest height.

        Returns:
            (trajectory, joint_mask)
        """
        if receiver_pos is None:
            receiver_pos = (0.45, 0.0, 1.05)  # forward at chest height

        return self.plan_reach(receiver_pos, arm, num_steps=num_steps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set(self, ctrl: np.ndarray, joint_name: str, value: float) -> None:
        """Set a control value for a named joint if it exists."""
        idx = self._idx.get(joint_name)
        if idx is not None:
            ctrl[idx] = float(value)

    @property
    def right_arm_mask(self) -> np.ndarray:
        return self._mask_right.copy()

    @property
    def left_arm_mask(self) -> np.ndarray:
        return self._mask_left.copy()

    @property
    def arms_mask(self) -> np.ndarray:
        return self._mask_arms.copy()


# ---------------------------------------------------------------------------
# HeadController
# ---------------------------------------------------------------------------


class HeadController:
    """Simple head/gaze controller.

    H1 has no dedicated neck joints, so this controller uses the torso
    joint (index 10) for small gaze-direction adjustments.  For models
    with neck pan/tilt actuators it generates appropriate joint targets.

    Parameters
    ----------
    joint_names:
        Ordered list of actuator names.
    standing_qpos:
        Standing pose q_des (19,).
    """

    def __init__(
        self,
        joint_names: list[str],
        standing_qpos: np.ndarray | None = None,
    ) -> None:
        self._joint_names = list(joint_names)
        self._n_joints = len(joint_names)
        self._has_neck = any("neck" in n.lower() for n in joint_names)

        if standing_qpos is not None:
            self._standing_qpos = np.asarray(standing_qpos, dtype=np.float64)
        else:
            self._standing_qpos = np.zeros(self._n_joints, dtype=np.float64)

        self._idx: dict[str, int] = {}
        for i, name in enumerate(self._joint_names):
            self._idx[name.replace("_act", "")] = i

        # Build dynamic masks
        _masks = make_joint_masks(self._n_joints)
        self._mask_torso = _masks.get("torso", _masks["none"])
        self._mask_none = _masks["none"]

    def plan_gaze(
        self,
        target_xyz: tuple[float, float, float],
        num_steps: int = 10,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate torso/head joint targets to look at a point.

        For H1 (no neck), uses a small torso rotation to orient the
        upper body toward the target.

        Returns:
            (trajectory, self._mask_torso)  — or (trajectory, self._mask_none) if
            no suitable joints exist.
        """
        if self._has_neck:
            # Model has neck joints — compute pan/tilt
            tx, ty, tz = target_xyz
            pan = math.atan2(ty, tx) * 0.3
            tilt = math.atan2(tz - 1.5, abs(tx)) * 0.3

            trajectory: list[np.ndarray] = []
            start = self._standing_qpos.copy()
            end = self._standing_qpos.copy()
            for name in self._joint_names:
                jname = name.replace("_act", "")
                if "pan" in jname.lower():
                    self._set(end, jname, pan)
                elif "tilt" in jname.lower():
                    self._set(end, jname, tilt)

            trajectory = _interpolate_linear(start, end, max(1, num_steps))
            return trajectory, self._mask_torso

        # No neck — small torso rotation toward target
        tx, ty, tz = target_xyz
        # Compute horizontal angle to target
        yaw_to_target = math.atan2(ty, tx) * 0.15  # small fraction

        trajectory: list[np.ndarray] = []
        start = self._standing_qpos.copy()
        end = self._standing_qpos.copy()
        torso_idx = self._idx.get("torso")
        if torso_idx is not None:
            end[torso_idx] = self._standing_qpos[torso_idx] + yaw_to_target

        trajectory = _interpolate_linear(start, end, max(1, num_steps))
        return trajectory, self._mask_torso

    def _set(self, ctrl: np.ndarray, joint_name: str, value: float) -> None:
        idx = self._idx.get(joint_name)
        if idx is not None:
            ctrl[idx] = float(value)


# ---------------------------------------------------------------------------
# StandingController (deprecated — replaced by BalanceController)
# ---------------------------------------------------------------------------


class StandingController:
    """Full-body standing balance controller for 19-DOF humanoids.

    .. deprecated::
        Use ``BalanceController`` from ``roboclaw.sim.balance`` instead.
        This class is kept for backward compatibility with existing tests
        and code that references ``StandingController.compute_control()``.

        ``BalanceController`` handles torque computation, elastic band,
        gravity compensation, and COM feedback in one place, whereas
        ``StandingController`` was designed for position-servo actuators
        (gainprm=1), which does not apply to H1 (gaintype=0).
    """

    LEG_JOINTS = (
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
        "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
        "right_knee", "right_ankle",
    )
    ARM_JOINTS = (
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow",
    )
    TORSO_JOINT = "torso"
    ANKLE_LEFT = "left_ankle"
    ANKLE_RIGHT = "right_ankle"
    HIP_PITCH_LEFT = "left_hip_pitch"
    HIP_PITCH_RIGHT = "right_hip_pitch"

    def __init__(
        self,
        joint_names: list[str],
        default_pose: np.ndarray | None = None,
        kp: float = 60.0,
        kd: float = 8.0,
        com_kp: float = 15.0,
    ) -> None:
        self._joint_names = list(joint_names)
        self._n_joints = len(joint_names)
        self._kp = kp
        self._kd = kd
        self._com_kp = com_kp

        if default_pose is not None:
            self._default_pose = np.asarray(default_pose, dtype=np.float64)
        else:
            self._default_pose = np.zeros(self._n_joints, dtype=np.float64)

        self._joint_index: dict[str, int] = {}
        for i, name in enumerate(self._joint_names):
            self._joint_index[name] = i

        self._act_to_joint: dict[int, str] = {}
        for i, name in enumerate(self._joint_names):
            self._act_to_joint[i] = name.replace("_act", "")

        # Build dynamic mask
        _masks = make_joint_masks(self._n_joints)
        self._mask_none = _masks["none"]

    def compute_control(
        self,
        joint_positions: dict[str, float],
        joint_velocities: dict[str, float],
        com_position: list[float] | None = None,
        com_desired: list[float] | None = None,
        gravity_comp: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the control vector for the current state.

        .. note::
           This method was designed for position-servo actuators
           (gainprm=1, ``force = ctrl - q``).  For H1 (gaintype=0,
           ``force = ctrl``), use ``BalanceController.step_control()``
           instead.
        """
        ctrl = np.zeros(self._n_joints, dtype=np.float64)

        if com_desired is None:
            com_desired = [0.0, 0.0, 1.0]

        com_error_x = 0.0
        com_error_y = 0.0
        if com_position is not None and len(com_position) >= 2:
            com_error_x = com_position[0] - com_desired[0]
            com_error_y = com_position[1] - com_desired[1]

        for i in range(self._n_joints):
            joint_name = self._act_to_joint.get(i, "")
            q_des = self._default_pose[i]
            q = joint_positions.get(joint_name, q_des)
            qd = joint_velocities.get(joint_name, 0.0)

            ctrl[i] = q + self._kp * (q_des - q) - self._kd * qd

            if gravity_comp is not None and i < len(gravity_comp):
                ctrl[i] += float(gravity_comp[i])

            if joint_name == self.ANKLE_LEFT:
                ctrl[i] -= self._com_kp * com_error_x
            elif joint_name == self.ANKLE_RIGHT:
                ctrl[i] -= self._com_kp * com_error_x

            if joint_name == self.HIP_PITCH_LEFT:
                ctrl[i] -= self._com_kp * 0.5 * com_error_x
            elif joint_name == self.HIP_PITCH_RIGHT:
                ctrl[i] -= self._com_kp * 0.5 * com_error_x

        return ctrl

    def plan_stand(
        self,
        duration_sec: float = 5.0,
        control_dt: float = 0.01,
        snapshot_provider: Any = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a standing trajectory.

        Returns:
            (trajectory, self._mask_none) — all joints track standing pose.
        """
        num_steps = max(1, int(duration_sec / control_dt))
        trajectory: list[np.ndarray] = []

        for _ in range(num_steps):
            trajectory.append(self._default_pose.copy())

        return trajectory, self._mask_none

    @property
    def default_pose(self) -> np.ndarray:
        return self._default_pose.copy()

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)
