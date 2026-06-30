"""MuJoCo physics environment wrapper.

Provides a thin, async-safe wrapper around a MuJoCo ``mjModel``/``mjData``
pair.  The environment handles model loading, physics stepping, sensor
snapshot extraction, and optional rendering.

Designed to be shared across multiple executors (locomotion, manipulation,
gaze) via asyncio-based cooperative locking.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from roboclaw.sim.config import MujocoConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MuJoCo import guard
# ---------------------------------------------------------------------------

_mujoco_available: bool | None = None


def _check_mujoco() -> bool:
    """Return True if MuJoCo Python bindings are importable."""
    global _mujoco_available
    if _mujoco_available is None:
        try:
            import mujoco  # noqa: F401
            _mujoco_available = True
        except ImportError:
            _mujoco_available = False
    return _mujoco_available


# ---------------------------------------------------------------------------
# Minimal Snapshot helpers
# ---------------------------------------------------------------------------


def _compute_zmp(
    mj_model: Any,
    mj_data: Any,
    foot_geom_names: list[str] | None = None,
) -> tuple[float, float]:
    """Approximate ZMP from ground contact forces on foot geoms.

    Returns (zx, zy) in world coordinates.  If no foot contact is detected,
    returns the projection of the COM onto the ground plane.
    """
    total_fz = 0.0
    zx_sum = 0.0
    zy_sum = 0.0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        # Contact force in world frame: product of contact frame and force array
        force = np.zeros(6)
        mujoco_module = _get_mujoco()
        mujoco_module.mj_contactForce(mj_model, mj_data, i, force)
        fz = force[2]  # vertical component
        if fz > 1e-6:
            pos = contact.pos
            total_fz += fz
            zx_sum += fz * pos[0]
            zy_sum += fz * pos[1]

    if total_fz > 1e-6:
        return (zx_sum / total_fz, zy_sum / total_fz)

    # No ground contacts — project COM onto ground
    com = mj_data.subtree_com[0] if mj_data.subtree_com.shape[0] > 0 else np.zeros(3)
    return (float(com[0]), float(com[1]))


def _get_mujoco():
    """Lazy mujoco import — raises ImportError if not available."""
    import mujoco

    return mujoco


# ---------------------------------------------------------------------------
# MujocoEnvironment
# ---------------------------------------------------------------------------


class MujocoEnvironment:
    """Wraps a MuJoCo model and data pair with async-safe physics stepping.

    Parameters
    ----------
    model_path:
        Path to an MJCF or URDF model file.
    config:
        MuJoCo engine configuration.  Uses defaults when omitted.
    control_decimation:
        Number of physics steps per :meth:`step` call.  Default 1.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: MujocoConfig | None = None,
        control_decimation: int = 1,
    ) -> None:
        if not _check_mujoco():
            raise ImportError(
                "MuJoCo is not installed. Run: pip install roboclaw[sim]"
            )
        mujoco = _get_mujoco()
        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._config = config or MujocoConfig()
        self._control_decimation = max(1, control_decimation)
        self._lock = asyncio.Lock()
        self._latest_snapshot: dict[str, Any] = {}
        self._joint_names: list[str] = []
        self._actuator_names: list[str] = []
        self._time: float = 0.0
        self._step_count: int = 0
        self._closed: bool = False

        # Viewer integration — when set, the env syncs the viewer
        # during :meth:`step` so physics changes are visible in real time.
        self._viewer: Any = None
        self._viewer_sync_every: int = 1  # sync every N calls to step()
        self._step_sync_counter: int = 0

        self._apply_config()
        self._build_name_maps()

    @staticmethod
    def available() -> bool:
        """Check whether MuJoCo is installed."""
        return _check_mujoco()

    @staticmethod
    def from_string(
        xml: str,
        config: MujocoConfig | None = None,
        control_decimation: int = 1,
    ) -> "MujocoEnvironment":
        """Create an environment from an in-memory MJCF XML string.

        Useful for unit tests that should not depend on external model files.
        """
        mujoco = _get_mujoco()
        model = mujoco.MjModel.from_xml_string(xml)
        return MujocoEnvironment._from_model(model, config, control_decimation)

    @staticmethod
    def _from_model(
        model: Any,
        config: MujocoConfig | None = None,
        control_decimation: int = 1,
        data: Any = None,
    ) -> "MujocoEnvironment":
        """Create an environment from an already-loaded ``mjModel``.

        When *data* is provided, it is shared (not copied) — use this
        to keep the MuJoCo state in sync with a ``BalanceController``
        or viewer that already holds a reference to the same ``mjData``.
        """
        mujoco = _get_mujoco()
        if data is None:
            data = mujoco.MjData(model)
        env = MujocoEnvironment.__new__(MujocoEnvironment)
        env._model = model
        env._data = data
        env._config = config or MujocoConfig()
        env._control_decimation = max(1, control_decimation)
        env._lock = asyncio.Lock()
        env._latest_snapshot: dict[str, Any] = {}
        env._joint_names: list[str] = []
        env._actuator_names: list[str] = []
        env._time: float = 0.0
        env._step_count: int = 0
        env._closed: bool = False
        env._viewer: Any = None
        env._viewer_sync_every: int = 1
        env._step_sync_counter: int = 0
        env._apply_config()
        env._build_name_maps()
        return env

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _apply_config(self) -> None:
        """Apply MujocoConfig to the mjModel options."""
        mujoco = _get_mujoco()
        opt = self._model.opt

        # Gravity
        g = self._config.gravity
        self._model.opt.gravity[:] = g

        # Solver
        solver_map = {"Newton": mujoco.mjtSolver.mjSOL_NEWTON,
                       "CG": mujoco.mjtSolver.mjSOL_CG,
                       "PGS": mujoco.mjtSolver.mjSOL_PGS}
        opt.solver = solver_map.get(self._config.solver, mujoco.mjtSolver.mjSOL_NEWTON)

        # Integrator
        integrator_map = {"Euler": mujoco.mjtIntegrator.mjINT_EULER,
                           "RK4": mujoco.mjtIntegrator.mjINT_RK4,
                           "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                           "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
        opt.integrator = integrator_map.get(
            self._config.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        )

        # Cone
        cone_map = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                     "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}
        opt.cone = cone_map.get(self._config.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL)

        # Jacobian
        jac_map = {"auto": 0, "dense": mujoco.mjtJacobian.mjJAC_DENSE,
                    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE}
        opt.jacobian = jac_map.get(self._config.jacobian, 0)

        opt.iterations = self._config.iterations
        opt.tolerance = self._config.tolerance

    def _build_name_maps(self) -> None:
        """Build ordered lists of joint and actuator names from the model."""
        mujoco = _get_mujoco()
        self._joint_names = []
        for jid in range(self._model.njnt):
            name_start = self._model.name_jntadr[jid]
            name = self._model.names[name_start:].decode("utf-8").split("\x00")[0]
            if name:  # skip anonymous joints
                self._joint_names.append(name)

        self._actuator_names = []
        for aid in range(self._model.nu):
            name_start = self._model.name_actuatoradr[aid]
            name = self._model.names[name_start:].decode("utf-8").split("\x00")[0]
            if name:
                self._actuator_names.append(name)

    # ------------------------------------------------------------------
    # Viewer integration
    # ------------------------------------------------------------------

    def set_viewer(self, viewer: Any, sync_every: int = 5) -> None:
        """Attach a MuJoCo passive viewer for real-time visual feedback.

        When set, :meth:`step` calls ``viewer.sync()`` every *sync_every*
        invocations so the rendered frames reflect the current physics
        state.  Set *sync_every* higher to reduce GPU load (e.g. 10 →
        syncs at ~100 Hz with a 1 kHz physics rate).

        Parameters
        ----------
        viewer:
            A ``mujoco.viewer`` handle (from ``launch_passive``).
        sync_every:
            Sync the viewer once every N calls to :meth:`step`.
        """
        self._viewer = viewer
        self._viewer_sync_every = max(1, sync_every)
        self._step_sync_counter = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """The underlying ``mjModel``."""
        return self._model

    @property
    def data(self) -> Any:
        """The underlying ``mjData`` (current state)."""
        return self._data

    @property
    def n_joints(self) -> int:
        """Number of joints in the model."""
        return self._model.njnt

    @property
    def n_actuators(self) -> int:
        """Number of actuators (control dimensions)."""
        return self._model.nu

    @property
    def n_qpos(self) -> int:
        """Degrees of freedom (positional)."""
        return self._model.nq

    @property
    def n_qvel(self) -> int:
        """Degrees of freedom (velocity)."""
        return self._model.nv

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of named joints."""
        return list(self._joint_names)

    @property
    def actuator_names(self) -> list[str]:
        """Ordered list of named actuators."""
        return list(self._actuator_names)

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @property
    def sim_time(self) -> float:
        """Alias for ``time``."""
        return self._time

    @property
    def step_count(self) -> int:
        """Number of physics steps executed."""
        return self._step_count

    @property
    def closed(self) -> bool:
        """True if ``close()`` has been called."""
        return self._closed

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def step(self, ctrl: np.ndarray | None = None) -> dict[str, Any]:
        """Advance the simulation by one control step.

        Applies the given control signal (matching actuator ordering),
        then steps the physics ``control_decimation`` times internally.
        Returns a sensor snapshot dict.

        Parameters
        ----------
        ctrl:
            Control array of shape ``(n_actuators,)`` or None to
            apply zero control.

        Returns
        -------
        dict
            Snapshot with keys: ``time``, ``qpos``, ``qvel``,
            ``joint_positions``, ``joint_velocities``, ``com_position``,
            ``zmp``, ``contact_forces``, ``imu``, ``step_count``.
        """
        if self._closed:
            raise RuntimeError("MujocoEnvironment is closed")

        async with self._lock:
            mujoco = _get_mujoco()

            # Apply control
            if ctrl is not None:
                ctrl_arr = np.asarray(ctrl, dtype=np.float64)
                if ctrl_arr.shape[0] != self._model.nu:
                    raise ValueError(
                        f"Control size mismatch: got {ctrl_arr.shape[0]}, "
                        f"expected {self._model.nu}"
                    )
                self._data.ctrl[:] = ctrl_arr
            else:
                self._data.ctrl[:] = 0.0

            # Step physics
            for _ in range(self._control_decimation):
                mujoco.mj_step(self._model, self._data)

            self._time += self._model.opt.timestep * self._control_decimation
            self._step_count += 1

            # Periodically sync the viewer so motion is visible in real time
            if self._viewer is not None and self._viewer.is_running():
                self._step_sync_counter += 1
                if self._step_sync_counter >= self._viewer_sync_every:
                    self._viewer.sync()
                    self._step_sync_counter = 0

            # Build snapshot
            snapshot = self._build_snapshot()
            self._latest_snapshot = snapshot

        return snapshot

    async def reset(self, qpos: np.ndarray | None = None) -> dict[str, Any]:
        """Reset the simulation to the initial (or given) joint configuration.

        Parameters
        ----------
        qpos:
            Optional joint position array.  Must match ``n_qpos``.
            When omitted, the model's ``qpos0`` is used.

        Returns
        -------
        dict
            Snapshot after reset.
        """
        async with self._lock:
            mujoco = _get_mujoco()
            if qpos is not None:
                qpos_arr = np.asarray(qpos, dtype=np.float64)
                if qpos_arr.shape[0] != self._model.nq:
                    raise ValueError(
                        f"qpos size mismatch: got {qpos_arr.shape[0]}, "
                        f"expected {self._model.nq}"
                    )
                self._data.qpos[:] = qpos_arr
            else:
                self._data.qpos[:] = self._model.qpos0

            self._data.qvel[:] = 0.0
            self._data.ctrl[:] = 0.0
            self._time = 0.0
            self._step_count = 0

            # Forward kinematics after reset
            mujoco.mj_forward(self._model, self._data)

            snapshot = self._build_snapshot()
            self._latest_snapshot = snapshot

        return snapshot

    def get_snapshot(self) -> dict[str, Any]:
        """Return the most recent sensor snapshot without stepping.

        Returns an empty dict if ``step()`` has never been called.
        """
        return dict(self._latest_snapshot)

    def close(self) -> None:
        """Release resources.  The environment is unusable after this."""
        self._closed = True
        logger.debug("MujocoEnvironment closed")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        """Render the current view and return an RGB pixel array.

        Requires a display / EGL context.  Returns None in headless mode.
        """
        if self._closed:
            return None
        try:
            mujoco = _get_mujoco()
            with self._model.opt.disableflags:
                pass  # ensure flags are in a context
            renderer = mujoco.Renderer(self._model, 480, 640)
            renderer.update_scene(self._data, camera="track")
            pixels = renderer.render()
            renderer.close()
            return pixels
        except Exception:
            logger.debug("Render not available (headless mode)", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> dict[str, Any]:
        """Extract a comprehensive sensor snapshot from the current state."""
        data = self._data
        model = self._model

        # Joint positions (named)
        joint_positions: dict[str, float] = {}
        joint_velocities: dict[str, float] = {}
        for jid in range(model.njnt):
            name_start = model.name_jntadr[jid]
            name = model.names[name_start:].decode("utf-8").split("\x00")[0]
            if not name:
                continue
            qpos_addr = model.jnt_qposadr[jid]
            qvel_addr = model.jnt_dofadr[jid]
            joint_type = model.jnt_type[jid]
            mujoco = _get_mujoco()
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                # Free joint: 7 qpos (pos + quat), 6 qvel
                joint_positions[f"{name}_x"] = float(data.qpos[qpos_addr])
                joint_positions[f"{name}_y"] = float(data.qpos[qpos_addr + 1])
                joint_positions[f"{name}_z"] = float(data.qpos[qpos_addr + 2])
                joint_positions[f"{name}_qw"] = float(data.qpos[qpos_addr + 3])
                joint_positions[f"{name}_qx"] = float(data.qpos[qpos_addr + 4])
                joint_positions[f"{name}_qy"] = float(data.qpos[qpos_addr + 5])
                joint_positions[f"{name}_qz"] = float(data.qpos[qpos_addr + 6])
            elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                joint_positions[name] = float(data.qpos[qpos_addr])
                joint_velocities[name] = float(data.qvel[qvel_addr])
            else:
                joint_positions[name] = float(data.qpos[qpos_addr])

        # COM
        com = np.zeros(3)
        if data.subtree_com.shape[0] > 0:
            com = data.subtree_com[0].copy()

        # ZMP
        zmp = _compute_zmp(model, data)

        # Contact forces summary
        contact_forces: list[dict[str, Any]] = []
        total_contact_force = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            force = np.zeros(6)
            mujoco = _get_mujoco()
            mujoco.mj_contactForce(model, data, i, force)
            mag = float(np.linalg.norm(force[:3]))
            total_contact_force += mag
            contact_forces.append({
                "geom1": int(contact.geom1),
                "geom2": int(contact.geom2),
                "position": contact.pos.tolist(),
                "force": force[:3].tolist(),
                "magnitude": mag,
            })

        # IMU (from accelerometer/gyro sensors if present, else approximate)
        imu: dict[str, list[float]] = {
            "accelerometer": [0.0, 0.0, -9.81],
            "gyroscope": [0.0, 0.0, 0.0],
        }
        for sid in range(model.nsensor):
            sensor_type = model.sensor_type[sid]
            sensor_name_start = model.name_sensoradr[sid]
            sensor_name = model.names[sensor_name_start:].decode("utf-8").split("\x00")[0]
            sensor_adr = model.sensor_adr[sid]
            if sensor_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER and sensor_adr < data.sensordata.shape[0]:
                imu["accelerometer"] = data.sensordata[sensor_adr:sensor_adr + 3].tolist()
            elif sensor_type == mujoco.mjtSensor.mjSENS_GYRO and sensor_adr < data.sensordata.shape[0]:
                imu["gyroscope"] = data.sensordata[sensor_adr:sensor_adr + 3].tolist()

        # Body poses (key bodies only — torso, pelvis, feet, hands, ankles, wrists)
        body_poses: dict[str, Any] = {}
        # Use substring matching so we work with both simple test models
        # ("torso", "left_foot") and production models like H1
        # ("torso_link", "pelvis", "left_ankle_link", "left_elbow_link").
        _key_body_patterns = (
            "torso", "pelvis",
            "foot", "ankle", "toe",
            "hand", "wrist", "elbow", "shoulder",
            "hip", "knee",
        )
        for bid in range(model.nbody):
            name_start = model.name_bodyadr[bid]
            name = model.names[name_start:].decode("utf-8").split("\x00")[0]
            if not name or name == "world":
                continue
            if any(p in name for p in _key_body_patterns):
                pos = data.xpos[bid].copy()
                xmat = data.xmat[bid].reshape(3, 3).copy()
                body_poses[name] = {
                    "position": pos.tolist(),
                    "rotation": xmat.tolist(),
                }

        return {
            "time": self._time,
            "step_count": self._step_count,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "com_position": com.tolist(),
            "zmp": list(zmp),
            "contact_forces": contact_forces,
            "total_contact_force": total_contact_force,
            "imu": imu,
            "body_poses": body_poses,
        }
