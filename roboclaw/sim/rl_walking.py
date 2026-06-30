"""RL-based H1 locomotion using Unitree's pre-trained policy.

Replicates the exact control architecture from Unitree's
``deploy/deploy_mujoco/deploy_mujoco.py`` in `unitree_rl_gym
<https://github.com/unitreerobotics/unitree_rl_gym>`_:

* PD control at every physics step (500 Hz)
* Policy inference every ``control_decimation`` steps (50 Hz)
* Observation: angular_vel + gravity + cmd + joint_pos + joint_vel + prev_action + clock
* Action → unscaled joint position targets for 10 leg joints

The pre-trained policy (`deploy/pre_train/h1/motion.pt`) is a TorchScript
model exported from PPO training in Isaac Gym.  It controls only the
lower body (10 DOF); the upper body (torso + arms, 9 DOF) must be
stabilised separately.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unitree H1 RL walking configuration (from h1.yaml)
# ---------------------------------------------------------------------------

# Default joint angles (radians) for the 10-DOF lower body.
# Order: left_hip_yaw(0), left_hip_roll(1), left_hip_pitch(2),
#        left_knee(3), left_ankle(4),
#        right_hip_yaw(5), right_hip_roll(6), right_hip_pitch(7),
#        right_knee(8), right_ankle(9)
DEFAULT_ANGLES = np.array(
    [0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, -0.1, 0.3, -0.2],
    dtype=np.float32,
)

# PD stiffness (N·m/rad)
KP_RL = np.array(
    [150.0, 150.0, 150.0, 200.0, 40.0,
     150.0, 150.0, 150.0, 200.0, 40.0],
    dtype=np.float32,
)

# PD damping (N·m·s/rad)
KD_RL = np.array(
    [2.0, 2.0, 2.0, 4.0, 2.0,
     2.0, 2.0, 2.0, 4.0, 2.0],
    dtype=np.float32,
)

# Scaling factors for observation normalisation
ACTION_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Control timing (from h1.yaml: dt=0.002, decimation=10 → 50 Hz policy)
CONTROL_DECIMATION = 10

# Observation / action dimensions
NUM_OBS = 41
NUM_ACTIONS = 10

# Default velocity command: 0.5 m/s forward
DEFAULT_CMD = np.array([0.5, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Gravity projection (exact copy from Unitree's deploy_mujoco.py)
# ---------------------------------------------------------------------------

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Project gravity vector into the robot's body frame.

    Parameters
    ----------
    quaternion:
        Base orientation quaternion ``[qw, qx, qy, qz]`` (MuJoCo convention).

    Returns
    -------
    gravity:
        Gravity direction in body frame, shape ``(3,)``.
    """
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Position-velocity PD torque computation (identical to Unitree's)."""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ---------------------------------------------------------------------------
# RL Walking Policy
# ---------------------------------------------------------------------------


class RLWalkingPolicy:
    """Load and run Unitree's pre-trained H1 locomotion policy.

    This class wraps a TorchScript policy exported from PPO training and
    provides :meth:`step` to run one policy inference cycle (observation
    → action → unscaled joint targets).

    Parameters
    ----------
    model_path:
        Path to the ``motion.pt`` TorchScript model.
    device:
        PyTorch device string (default ``"cpu"`` — fast enough at ~0.4 ms/inference).
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._policy: torch.jit.ScriptModule = torch.jit.load(model_path)
        self._policy.eval()
        self._device = device

        # Persistent state across policy calls
        self._prev_action: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._step_counter: int = 0

        logger.info(
            "RLWalkingPolicy loaded: %s (inference ~%.2f ms on %s)",
            model_path, 0.4, device,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset persistent policy state (previous action and step counter)."""
        self._prev_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._step_counter = 0

    # ------------------------------------------------------------------
    def compute_observation(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        cmd: np.ndarray,
    ) -> np.ndarray:
        """Build the 41-dim observation vector from MuJoCo state.

        Parameters
        ----------
        qpos:
            Joint positions for the 10 leg actuators.
        qvel:
            Joint velocities for the 10 leg actuators.
        quat:
            Base orientation quaternion ``[qw, qx, qy, qz]``.
        omega:
            Base angular velocity (body frame).
        cmd:
            Commanded velocity ``[vx, vy, vyaw]`` in world frame.

        Returns
        -------
        obs:
            Normalised observation vector, shape ``(41,)``.
        """
        # Centre and scale proprioception
        qj = (qpos - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj = qvel * DOF_VEL_SCALE
        gravity = get_gravity_orientation(quat)
        omega_scaled = omega * ANG_VEL_SCALE

        # Phase clock (0.8 s gait period, driven by wall-clock time)
        period = 0.8
        count = self._step_counter * 0.002  # sim_dt = 0.002 s
        phase = (count % period) / period
        sin_phase = np.sin(2.0 * np.pi * phase)
        cos_phase = np.cos(2.0 * np.pi * phase)

        # Assemble observation (exact layout from deploy_mujoco.py)
        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[:3] = omega_scaled
        obs[3:6] = gravity
        obs[6:9] = cmd * CMD_SCALE
        obs[9:19] = qj
        obs[19:29] = dqj
        obs[29:39] = self._prev_action
        obs[39:41] = [sin_phase, cos_phase]

        return obs

    # ------------------------------------------------------------------
    def step(self, data: Any, cmd: np.ndarray) -> np.ndarray:
        """Run one policy inference and return unscaled leg joint targets.

        Parameters
        ----------
        data:
            MuJoCo ``MjData`` instance.
        cmd:
            Commanded velocity ``[vx, vy, vyaw]``.

        Returns
        -------
        target_q:
            Absolute joint position targets for the 10 leg joints.
        """
        # Extract state from MuJoCo data
        qpos = data.qpos[7:17].copy()   # first 10 actuators
        qvel = data.qvel[6:16].copy()   # first 10 actuators
        quat = data.qpos[3:7].copy()    # base orientation
        omega = data.qvel[3:6].copy()   # base angular velocity

        obs = self.compute_observation(qpos, qvel, quat, omega, cmd)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self._device)

        with torch.no_grad():
            action = self._policy(obs_tensor).detach().cpu().numpy().squeeze()

        self._prev_action = action.copy()
        self._step_counter += CONTROL_DECIMATION  # counter tracks simulation steps

        # Unscale action → absolute joint position targets
        target_q = DEFAULT_ANGLES + ACTION_SCALE * action
        return target_q.astype(np.float64)

    # ------------------------------------------------------------------
    @property
    def step_counter(self) -> int:
        return self._step_counter


# ---------------------------------------------------------------------------
# RL walking execution helper
# ---------------------------------------------------------------------------


def run_rl_walking_step(
    data: Any,
    target_dof_pos_legs: np.ndarray,
    standing_qpos_upper: np.ndarray,
    kp_upper: np.ndarray,
    kd_upper: np.ndarray,
) -> np.ndarray:
    """Compute PD torque for one physics step during RL walking.

    Legs (indices 0-9) are driven to *target_dof_pos_legs* using the
    RL-trained PD gains.  Upper body (indices 10+) tracks the standing
    pose using the BalanceController's PD gains.

    Parameters
    ----------
    data:
        MuJoCo ``MjData``.
    target_dof_pos_legs:
        Joint position targets for legs, shape ``(10,)``.
    standing_qpos_upper:
        Standing pose for torso + arms, shape ``(n_act-10,)``.
    kp_upper:
        PD stiffness for upper body joints, shape ``(n_act-10,)``.
    kd_upper:
        PD damping for upper body joints, shape ``(n_act-10,)``.

    Returns
    -------
    ctrl:
        Torque control vector, shape ``(n_act,)`` (all actuators).
    """
    n_act = len(data.ctrl)
    q_offset = data.qpos.shape[0] - n_act if data.qpos.shape[0] > n_act else 0
    v_offset = q_offset - 1 if q_offset > 0 else 0

    q = data.qpos[q_offset:q_offset + n_act].copy()
    qd = data.qvel[v_offset:v_offset + n_act].copy() if v_offset >= 0 else np.zeros(n_act)

    ctrl = np.zeros(n_act, dtype=np.float64)

    # Legs: RL-trained PD gains (up to min(10, n_act) joints)
    n_legs = min(10, n_act)
    leg_q_des = np.asarray(target_dof_pos_legs[:n_legs], dtype=np.float64)
    ctrl[:n_legs] = KP_RL[:n_legs] * (leg_q_des - q[:n_legs]) - KD_RL[:n_legs] * qd[:n_legs]

    # Upper body: track standing pose with BalanceController PD gains
    n_upper = n_act - n_legs
    if n_upper > 0:
        upper_q_des = np.asarray(standing_qpos_upper[:n_upper], dtype=np.float64)
        ctrl[n_legs:n_act] = (
            kp_upper[:n_upper] * (upper_q_des - q[n_legs:n_act])
            - kd_upper[:n_upper] * qd[n_legs:n_act]
        )

    return ctrl
