"""MuJoCo-based action executors with balance-aware physics stepping.

Every executor delegates torque computation to ``BalanceController``,
which provides elastic band support, gravity compensation, PD control,
and COM feedback at every simulation step.  Skill controllers generate
q_des trajectories + joint_mask — the executor only handles the
physics loop.
"""

from __future__ import annotations

import asyncio
import logging
import time as time_mod
from typing import Any

import numpy as np

from roboclaw.action.base import AbstractActionExecutor, ActionResult, ActionResultStatus
from roboclaw.sim.controllers import WalkingController, GraspingController, HeadController
from roboclaw.sim.mujoco_env import MujocoEnvironment
from roboclaw.sim.rl_walking import (
    RLWalkingPolicy,
    DEFAULT_ANGLES,
    KP_RL,
    KD_RL,
    CONTROL_DECIMATION,
    run_rl_walking_step,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_actual_outcome(
    skill_type: str,
    params: dict[str, Any],
    duration_sec: float,
    traj_len: int,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the ``actual_outcome`` dict for an ActionResult.

    All values are converted to native Python types to ensure
    msgpack/json serializability for LangGraph checkpoints.
    """
    final_com: list[float] = []
    final_zmp: list[float] = []
    final_body_poses: dict[str, Any] = {}
    sim_time: float = 0.0

    if snapshots:
        s = snapshots[-1]
        # Convert numpy scalars to native Python types
        final_com = [float(x) for x in s.get("com_position", [])]
        final_zmp = [float(x) for x in s.get("zmp", [])]
        final_body_poses = _ensure_native_dict(s.get("body_poses", {}))
        sim_time = float(s.get("time", 0.0))

    # Convert params values to native types (defensive)
    safe_params = _ensure_native_dict(params)

    outcome: dict[str, Any] = {
        "simulated": True,
        "backend": "mujoco",
        "skill": skill_type,
        "params": safe_params,
        "duration_sec": float(duration_sec),
        "control_steps": int(traj_len),
        "sim_time": sim_time,
        "final_com": final_com,
        "final_zmp": final_zmp,
        "trajectory_length": int(traj_len),
    }
    if snapshots:
        outcome["final_body_poses"] = final_body_poses
    return outcome


def _ensure_native_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert numpy types in a dict to native Python types."""
    import numpy as np

    result: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, (np.floating,)):
            result[k] = float(v)
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = _ensure_native_dict(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [
                float(x) if isinstance(x, (np.floating,)) else
                int(x) if isinstance(x, (np.integer,)) else x
                for x in v
            ]
        else:
            result[k] = v
    return result


async def _execute_with_balance(
    balance: Any,
    env: MujocoEnvironment,
    trajectory: list[np.ndarray],
    joint_mask: np.ndarray,
    ros_publisher: Any = None,
    cancel_flag: Any = None,
    pelvis_push: tuple[float, float, float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], bool]:
    """Execute a q_des trajectory with balance-aware torque control.

    At each step, calls ``balance.step_control(env, skill_q_des, joint_mask)``
    to compute the torque control vector (gravity comp + PD + elastic band
    + COM feedback), then steps the physics.

    Parameters
    ----------
    balance:
        ``BalanceController`` instance.
    env:
        The shared MuJoCo environment.
    trajectory:
        List of q_des arrays (each shape ``(19,)``).
    joint_mask:
        Float mask (19,) indicating which joints this skill controls.
    ros_publisher:
        Optional ROS 2 publisher for state streaming.
    cancel_flag:
        Optional list containing a single bool ``[False]`` that can be
        set to ``True`` from another coroutine to cancel execution.
    pelvis_push:
        Optional (fx, fy, fz) force applied to the pelvis every step.
        Used for locomotion propulsion (forward/backward/lateral).

    Returns
    -------
    (executed, snapshots, cancelled)
    """
    snapshots: list[dict[str, Any]] = []
    executed: list[np.ndarray] = []
    cancelled = False

    for i, skill_q_des in enumerate(trajectory):
        if cancel_flag is not None and cancel_flag[0]:
            cancelled = True
            break

        # Compute torque via shared balance controller
        ctrl = balance.step_control(
            env,
            skill_q_des=skill_q_des,
            joint_mask=joint_mask,
            external_force=pelvis_push,
        )

        # Step physics
        snapshot = await env.step(ctrl)
        snapshots.append(snapshot)
        executed.append(ctrl)

        # Publish to ROS 2 if bridge is attached
        if ros_publisher is not None:
            ros_publisher.publish_snapshot(snapshot)

        # Yield control to event loop periodically
        if i % 10 == 0:
            await asyncio.sleep(0)

    return executed, snapshots, cancelled


# ---------------------------------------------------------------------------
# MujocoLocomotionExecutor
# ---------------------------------------------------------------------------


class MujocoLocomotionExecutor(AbstractActionExecutor):
    """Executor for bipedal locomotion skills using MuJoCo physics.

    Skills handled: navigate_to, walk_steps, climb_stairs

    Parameters
    ----------
    env:
        The shared MuJoCo environment.
    balance:
        ``BalanceController`` for torque control + elastic band.
    robot_config:
        Robot configuration for skill defaults.
    controller:
        Optional pre-built WalkingController.
    """

    def __init__(
        self,
        env: MujocoEnvironment,
        balance: Any,
        robot_config: Any = None,
        controller: WalkingController | None = None,
        ros_publisher: Any = None,
        rl_policy: RLWalkingPolicy | None = None,
    ) -> None:
        self._env = env
        self._balance = balance
        self._config = robot_config
        self._cancelled = [False]  # mutable flag for cancellation
        self._ros_publisher = ros_publisher
        self._rl_policy = rl_policy

        if controller is not None:
            self._controller = controller
        else:
            self._controller = WalkingController(
                joint_names=env.actuator_names,
                standing_qpos=balance.standing_q_des if balance else None,
                step_height=getattr(
                    getattr(robot_config, 'skills', None), 'walk_step_height_m', 0.05
                ) if robot_config else 0.05,
                max_speed=getattr(
                    getattr(robot_config, 'skills', None), 'walk_max_speed_ms', 1.5
                ) if robot_config else 1.5,
            )

        # Cache PD gains for upper body (standing pose tracking)
        n_act = env._model.nu
        n_upper = max(0, n_act - 10)
        self._kp_upper = np.full(n_upper, balance.kp if balance else 150.0, dtype=np.float64)
        self._kd_upper = np.full(n_upper, balance.kd if balance else 15.0, dtype=np.float64)
        # Only enable RL for H1-like models (19 actuators, including 10 leg joints).
        # Smaller test models use the CPG fallback.
        self._has_rl = rl_policy is not None and n_act >= 19
        if self._has_rl:
            logger.info("MujocoLocomotionExecutor: RL walking policy ENABLED")

    # ------------------------------------------------------------------
    # AbstractActionExecutor interface
    # ------------------------------------------------------------------

    async def execute(
        self, sub_task: dict[str, Any], world_state: dict[str, Any]
    ) -> ActionResult:
        self._cancelled[0] = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "")
        sub_task_id = sub_task.get("sub_task_id", skill_type)

        try:
            if skill_type == "navigate_to":
                return await self._navigate_to(params, sub_task_id)
            elif skill_type == "walk_steps":
                return await self._walk_steps(params, sub_task_id)
            elif skill_type == "climb_stairs":
                return await self._climb_stairs(params, sub_task_id)
            else:
                return ActionResult(
                    sub_task_id=sub_task_id,
                    status=ActionResultStatus.FAILURE,
                    error={
                        "type": "unknown_skill",
                        "detail": f"Locomotion skill '{skill_type}' not supported",
                    },
                )
        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled", "detail": "Locomotion cancelled"},
            )

    async def cancel(self) -> None:
        self._cancelled[0] = True
        logger.info("MujocoLocomotionExecutor cancelled gracefully")

    async def emergency_stop(self) -> None:
        self._cancelled[0] = True
        logger.warning("MujocoLocomotionExecutor emergency stop!")

    async def _settle(self, steps: int = 50) -> None:
        """Run standing balance for *steps* to stabilize after locomotion."""
        import mujoco
        for _ in range(steps):
            ctrl = self._balance.step_control(self._env)
            self._env._data.ctrl[:] = ctrl
            mujoco.mj_step(self._env._model, self._env._data)

    # ------------------------------------------------------------------
    # Skill implementations
    # ------------------------------------------------------------------

    async def _navigate_to(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        room = params.get("room", "unknown")

        room_distances = {
            "kitchen": 1.5,
            "living_room": 1.2,
            "bedroom": 1.2,
            "doorway": 1.0,
        }
        distance = room_distances.get(
            str(room).lower(), 0.8 if isinstance(room, str) else 1.0,
        )

        # Use RL policy when available — 0.3 m/s is more stable than 0.5
        if self._has_rl:
            return await self._run_rl_walking(
                sub_task_id, "navigate_to", params,
                cmd_vel=(0.3, 0.03, 0.0),  # slight right correction for lateral drift
                distance=distance,
            )

        # Fallback: CPG trajectory + push force
        traj, mask = self._controller.plan_navigate(distance)
        result = await self._run_trajectory(
            traj, mask, sub_task_id, "navigate_to", params,
            pelvis_push=(80.0, 0.0, 0.0),
        )
        if result.status == ActionResultStatus.SUCCESS:
            await self._settle(steps=100)
        return result

    async def _walk_steps(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        num_steps: int = int(params.get("num_steps", 1))
        direction: str = str(params.get("direction", "forward"))
        step_length: float = float(params.get("step_length_m", 0.3))

        # Use RL policy when available — no external push forces needed
        if self._has_rl:
            # Map direction to velocity command
            cmd_map = {
                "forward": (0.3, 0.03, 0.0),   # slight right correction for lateral drift
                "backward": (-0.3, 0.0, 0.0),
                "left": (0.0, 0.3, 0.0),
                "right": (0.0, -0.3, 0.0),
            }
            cmd_vel = cmd_map.get(direction, (0.3, 0.03, 0.0))
            # Each step takes ~0.8 s at 0.5 m/s, step length ~0.4 m
            distance = num_steps * step_length
            return await self._run_rl_walking(
                sub_task_id, "walk_steps", params,
                cmd_vel=cmd_vel,
                distance=distance,
            )

        # Fallback: CPG trajectory + push force
        traj, mask = self._controller.plan_steps(num_steps, direction, step_length)
        scale = step_length / 0.3
        push_map = {
            "forward": (100.0 * scale, 0.0, 0.0),
            "backward": (-180.0 * scale, 0.0, 0.0),
            "left": (0.0, 100.0 * scale, 0.0),
            "right": (0.0, -100.0 * scale, 0.0),
        }
        pelvis_push = push_map.get(direction, (100.0 * scale, 0.0, 0.0))
        result = await self._run_trajectory(
            traj, mask, sub_task_id, "walk_steps", params,
            pelvis_push=pelvis_push,
        )
        if result.status == ActionResultStatus.SUCCESS:
            await self._settle(steps=80)
        return result

    async def _climb_stairs(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        num_stairs: int = int(params.get("num_steps", 5))
        direction: str = str(params.get("direction", "up"))

        traj, mask = self._controller.plan_climb(num_stairs, direction)
        # Gentle forward + upward assist for stair climbing.
        # Higher forces cause instability on the incline.
        push_z = 30.0 if direction == "up" else -15.0
        result = await self._run_trajectory(
            traj, mask, sub_task_id, "climb_stairs", params,
            pelvis_push=(40.0, 0.0, push_z),
        )
        if result.status == ActionResultStatus.SUCCESS:
            await self._settle(steps=60)
        return result

    # ------------------------------------------------------------------
    # RL walking execution
    # ------------------------------------------------------------------

    async def _run_rl_walking(
        self,
        sub_task_id: str,
        skill_type: str,
        params: dict[str, Any],
        cmd_vel: tuple[float, float, float],
        distance: float = 1.0,
    ) -> ActionResult:
        """Execute locomotion via Unitree's RL walking policy.

        Replicates ``deploy_mujoco.py`` from unitree_rl_gym:
        PD control at every physics step, policy inference every
        ``CONTROL_DECIMATION`` steps.
        """
        import mujoco

        if self._rl_policy is None:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.FAILURE,
                error={"type": "no_rl_policy",
                       "detail": "RL policy not loaded"},
            )

        self._rl_policy.reset()

        cmd = np.array(cmd_vel, dtype=np.float32)
        velocity = float(np.linalg.norm(cmd[:2]))
        if velocity < 0.01:
            velocity = 0.5

        # Duration in simulation steps (dt=0.002 s), capped at 3.0 s for stability
        duration_s = min(max(distance / velocity, 0.5), 3.0)
        total_steps = int(duration_s / 0.002)

        target_dof_pos_legs = DEFAULT_ANGLES.copy()
        standing_upper = self._balance.standing_qpos[10:19].copy()

        _model = self._env._model
        _data = self._env._data

        snapshots: list[dict[str, Any]] = []
        start_time = time_mod.time()
        cancelled = False

        step = 0
        while step < total_steps:
            if self._cancelled[0]:
                cancelled = True
                break

            # PD torque for all joints (legs via RL, upper body via standing)
            ctrl = run_rl_walking_step(
                _data, target_dof_pos_legs,
                standing_upper,
                self._kp_upper, self._kd_upper,
            )
            _data.ctrl[:] = ctrl
            mujoco.mj_step(_model, _data)
            step += 1

            # Policy inference every CONTROL_DECIMATION steps (50 Hz)
            if step % CONTROL_DECIMATION == 0:
                target_dof_pos_legs = self._rl_policy.step(_data, cmd)

            # Snapshot every 50 steps
            if step % 50 == 0:
                snapshots.append({
                    "step": step,
                    "time": float(_data.time),
                    "pelvis_xy": _data.xpos[self._balance.pelvis_id][:2].tolist(),
                    "pelvis_z": float(_data.xpos[self._balance.pelvis_id][2]),
                    "com_position": (
                        _data.subtree_com[0][:3].tolist()
                        if _data.subtree_com.shape[0] > 0
                        else [0.0, 0.0, 0.0]
                    ),
                })

            # Yield to event loop periodically
            if step % 100 == 0:
                await asyncio.sleep(0)

        # --- Zero-velocity cooldown: bring robot to a stop smoothly ---
        zero_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(150):  # 300 ms cooldown
            if self._cancelled[0]:
                break
            if step % CONTROL_DECIMATION == 0:
                target_dof_pos_legs = self._rl_policy.step(_data, zero_cmd)
            ctrl = run_rl_walking_step(
                _data, target_dof_pos_legs,
                standing_upper,
                self._kp_upper, self._kd_upper,
            )
            _data.ctrl[:] = ctrl
            mujoco.mj_step(_model, _data)
            step += 1

        duration = time_mod.time() - start_time

        # Compute actual displacement
        final_pelvis = _data.xpos[self._balance.pelvis_id][:2].tolist()
        initial_pelvis = snapshots[0]["pelvis_xy"] if snapshots else [0.0, 0.0]
        dx = final_pelvis[0] - initial_pelvis[0]
        dy = final_pelvis[1] - initial_pelvis[1]

        status = ActionResultStatus.INTERRUPTED if cancelled else ActionResultStatus.SUCCESS

        outcome = _build_actual_outcome(
            skill_type, params, duration, step, snapshots,
        )
        outcome["rl_walking"] = True
        outcome["displacement_xy"] = [float(dx), float(dy)]
        outcome["cmd_vel"] = [float(c) for c in cmd]

        logger.info(
            "RL walking: %s, %d steps (incl cooldown), cmd=%s, dx=%.3f m, dy=%.3f m",
            skill_type, step, cmd_vel, dx, dy,
        )

        return ActionResult(
            sub_task_id=sub_task_id,
            status=status,
            duration_sec=duration,
            actual_outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Shared trajectory execution (CPG fallback)
    # ------------------------------------------------------------------

    async def _run_trajectory(
        self,
        trajectory: list[np.ndarray],
        joint_mask: np.ndarray,
        sub_task_id: str,
        skill_type: str,
        params: dict[str, Any],
        pelvis_push: tuple[float, float, float] | None = None,
    ) -> ActionResult:
        if not trajectory:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.FAILURE,
                error={"type": "empty_trajectory",
                       "detail": "No control steps generated"},
            )

        start_time = time_mod.time()
        try:
            executed, snapshots, cancelled = await _execute_with_balance(
                self._balance, self._env, trajectory, joint_mask,
                ros_publisher=self._ros_publisher,
                cancel_flag=self._cancelled,
                pelvis_push=pelvis_push,
            )
            duration = time_mod.time() - start_time

            status = ActionResultStatus.INTERRUPTED if cancelled else ActionResultStatus.SUCCESS

            return ActionResult(
                sub_task_id=sub_task_id,
                status=status,
                duration_sec=duration,
                actual_outcome=_build_actual_outcome(
                    skill_type, params, duration, len(executed), snapshots
                ),
            )
        except Exception as exc:
            duration = time_mod.time() - start_time
            logger.error(f"Trajectory execution failed: {exc}")
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.FAILURE,
                duration_sec=duration,
                error={"type": "execution_error", "detail": str(exc)},
            )


# ---------------------------------------------------------------------------
# MujocoManipulationExecutor
# ---------------------------------------------------------------------------


class MujocoManipulationExecutor(AbstractActionExecutor):
    """Executor for dual-arm manipulation skills using MuJoCo physics.

    Skills handled: whole_body_grasp, place_object, handover, open_door

    Parameters
    ----------
    env:
        The shared MuJoCo environment.
    balance:
        ``BalanceController`` for torque control + elastic band.
    robot_config:
        Robot configuration for skill defaults.
    controller:
        Optional pre-built GraspingController.
    """

    def __init__(
        self,
        env: MujocoEnvironment,
        balance: Any,
        robot_config: Any = None,
        controller: GraspingController | None = None,
        ros_publisher: Any = None,
    ) -> None:
        self._env = env
        self._balance = balance
        self._config = robot_config
        self._cancelled = [False]
        self._ros_publisher = ros_publisher

        if controller is not None:
            self._controller = controller
        else:
            self._controller = GraspingController(
                joint_names=env.actuator_names,
                standing_qpos=balance.standing_q_des if balance else None,
                arm_workspace_radius=getattr(
                    getattr(robot_config, 'skills', None), 'arm_workspace_radius_m', 0.85
                ) if robot_config else 0.85,
            )

    # ------------------------------------------------------------------
    # AbstractActionExecutor interface
    # ------------------------------------------------------------------

    async def execute(
        self, sub_task: dict[str, Any], world_state: dict[str, Any]
    ) -> ActionResult:
        self._cancelled[0] = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "")
        sub_task_id = sub_task.get("sub_task_id", skill_type)

        try:
            if skill_type in ("whole_body_grasp",):
                return await self._grasp(params, sub_task_id)
            elif skill_type == "place_object":
                return await self._place(params, sub_task_id)
            elif skill_type == "handover":
                return await self._handover(params, sub_task_id)
            elif skill_type == "open_door":
                return await self._open_door(params, sub_task_id)
            else:
                return ActionResult(
                    sub_task_id=sub_task_id,
                    status=ActionResultStatus.FAILURE,
                    error={
                        "type": "unknown_skill",
                        "detail": f"Manipulation skill '{skill_type}' not supported",
                    },
                )
        except asyncio.CancelledError:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.INTERRUPTED,
                error={"type": "cancelled", "detail": "Manipulation cancelled"},
            )

    async def cancel(self) -> None:
        self._cancelled[0] = True
        logger.info("MujocoManipulationExecutor cancelled gracefully")

    async def emergency_stop(self) -> None:
        self._cancelled[0] = True
        logger.warning("MujocoManipulationExecutor emergency stop!")

    # ------------------------------------------------------------------
    # Skill implementations
    # ------------------------------------------------------------------

    async def _grasp(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        obj: str = str(params.get("object", "unknown"))
        arm: str = str(params.get("arm", "right"))
        grasp_type: str = str(params.get("grasp_type", "top_down_encompassing"))

        # Heuristic object position
        if arm == "left":
            obj_pos = (0.4, -0.2, 0.75)
        else:
            obj_pos = (0.4, 0.2, 0.75)

        traj, mask = self._controller.plan_grasp(obj_pos, arm, num_steps=50)
        return await self._run_trajectory(traj, mask, sub_task_id, "whole_body_grasp", params)

    async def _place(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        target = (
            float(params.get("target_x", 0.3)),
            float(params.get("target_y", 0.0)),
            float(params.get("target_z", 0.5)),
        )
        arm = str(params.get("arm", "right"))
        traj, mask = self._controller.plan_place(target, arm, num_steps=40)
        return await self._run_trajectory(traj, mask, sub_task_id, "place_object", params)

    async def _handover(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        arm = str(params.get("arm", "right"))
        traj, mask = self._controller.plan_handover(arm=arm, num_steps=60)
        return await self._run_trajectory(traj, mask, sub_task_id, "handover", params)

    async def _open_door(
        self, params: dict[str, Any], sub_task_id: str
    ) -> ActionResult:
        arm = str(params.get("arm", "right"))
        traj, mask = self._controller.plan_door_open(arm=arm, num_steps=80)
        return await self._run_trajectory(traj, mask, sub_task_id, "open_door", params)

    # ------------------------------------------------------------------
    # Shared trajectory execution
    # ------------------------------------------------------------------

    async def _run_trajectory(
        self,
        trajectory: list[np.ndarray],
        joint_mask: np.ndarray,
        sub_task_id: str,
        skill_type: str,
        params: dict[str, Any],
        pelvis_push: tuple[float, float, float] | None = None,
    ) -> ActionResult:
        if not trajectory:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.FAILURE,
                error={"type": "empty_trajectory",
                       "detail": "No control steps generated"},
            )

        start_time = time_mod.time()
        try:
            executed, snapshots, cancelled = await _execute_with_balance(
                self._balance, self._env, trajectory, joint_mask,
                ros_publisher=self._ros_publisher,
                cancel_flag=self._cancelled,
                pelvis_push=pelvis_push,
            )
            duration = time_mod.time() - start_time

            status = ActionResultStatus.INTERRUPTED if cancelled else ActionResultStatus.SUCCESS

            return ActionResult(
                sub_task_id=sub_task_id,
                status=status,
                duration_sec=duration,
                actual_outcome=_build_actual_outcome(
                    skill_type, params, duration, len(executed), snapshots
                ),
            )
        except Exception as exc:
            duration = time_mod.time() - start_time
            logger.error(f"Trajectory execution failed: {exc}")
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.FAILURE,
                duration_sec=duration,
                error={"type": "execution_error", "detail": str(exc)},
            )


# ---------------------------------------------------------------------------
# MujocoHeadNeckExecutor
# ---------------------------------------------------------------------------


class MujocoHeadNeckExecutor(AbstractActionExecutor):
    """Executor for head/gaze control using MuJoCo physics.

    Skills handled: gaze_at

    For H1 (no neck joints), uses small torso adjustments via the
    BalanceController.
    """

    def __init__(
        self,
        env: MujocoEnvironment,
        balance: Any,
        robot_config: Any = None,
        controller: HeadController | None = None,
    ) -> None:
        self._env = env
        self._balance = balance
        self._config = robot_config
        self._cancelled = [False]

        if controller is not None:
            self._controller = controller
        else:
            self._controller = HeadController(
                joint_names=env.actuator_names,
                standing_qpos=balance.standing_q_des if balance else None,
            )

    async def execute(
        self, sub_task: dict[str, Any], world_state: dict[str, Any]
    ) -> ActionResult:
        self._cancelled[0] = False
        params = sub_task.get("parameters", {})
        skill_type = sub_task.get("skill_type", "gaze_at")
        sub_task_id = sub_task.get("sub_task_id", skill_type)

        target = params.get("target", "unknown")
        # Heuristic target position
        target_xyz = (1.0, 0.0, 1.5)
        if isinstance(target, dict):
            target_xyz = (
                float(target.get("x", 1.0)),
                float(target.get("y", 0.0)),
                float(target.get("z", 1.5)),
            )

        traj, mask = self._controller.plan_gaze(target_xyz, num_steps=10)

        start_time = time_mod.time()
        try:
            executed, snapshots, cancelled = await _execute_with_balance(
                self._balance, self._env, traj, mask,
                cancel_flag=self._cancelled,
            )
            duration = time_mod.time() - start_time

            status = ActionResultStatus.INTERRUPTED if cancelled else ActionResultStatus.SUCCESS

            return ActionResult(
                sub_task_id=sub_task_id,
                status=status,
                duration_sec=duration,
                actual_outcome={
                    "simulated": True,
                    "backend": "mujoco",
                    "skill": skill_type,
                    "target": target_xyz,
                    "duration_sec": duration,
                    "control_steps": len(executed),
                    "sim_time": float(self._env.sim_time) if self._env else 0.0,
                },
            )
        except Exception as exc:
            return ActionResult(
                sub_task_id=sub_task_id,
                status=ActionResultStatus.FAILURE,
                error={"type": "execution_error", "detail": str(exc)},
            )

    async def cancel(self) -> None:
        self._cancelled[0] = True
        logger.info("MujocoHeadNeckExecutor cancelled")

    async def emergency_stop(self) -> None:
        self._cancelled[0] = True
        logger.warning("MujocoHeadNeckExecutor emergency stop!")
