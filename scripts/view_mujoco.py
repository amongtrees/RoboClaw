#!/usr/bin/env python3
"""Launch MuJoCo viewer with H1 standing balance via elastic band + PD + gravity comp.

H1 cannot self-balance from a pure PD controller — Unitree's official simulator
uses a virtual "elastic band" to provide vertical support during initialization.
We replicate that strategy here.

Close the viewer window to exit (or Ctrl+C).

Usage:
    conda activate roboclaw
    python -u scripts/view_mujoco.py
"""

from __future__ import annotations

import signal
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "models/mujoco_menagerie/unitree_h1/scene.xml"

_running = True


def _on_sigint(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _on_sigint)


# ---------------------------------------------------------------------------
def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # --- standing pose from home keyframe ----------------------------------
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    standing_qpos = data.qpos[7:].copy()

    # Adjust pelvis height so feet touch the ground
    _lower_to_ground(model, data)
    # Re-read standing pose after adjustment
    standing_qpos = data.qpos[7:].copy()
    pelvis_target_z = float(data.qpos[2])
    mujoco.mj_forward(model, data)

    # Pre-compute gravity torques at the adjusted standing pose
    g_comp_ref = data.qfrc_bias[6:].copy()

    # --- robot properties --------------------------------------------------
    total_mass = _compute_total_mass(model)
    pelvis_id = model.body("pelvis").id
    weight = total_mass * 9.81

    # --- controller parameters ---------------------------------------------
    kp, kd = 150.0, 15.0           # joint PD gains
    band_ratio = 0.65               # fraction of weight supported by band
    band_kp_xy = 500.0              # horizontal centering stiffness
    band_kd_xy = 100.0              # horizontal centering damping

    print(f"H1: {model.nbody} bodies, {model.nu} actuators, mass={total_mass:.1f} kg")
    print(f"Band support: {band_ratio*100:.0f}% weight ({band_ratio*weight:.0f} N)")
    print(f"Standing pose loaded. Close window or Ctrl+C to exit.\n")

    step_count = 0
    last_print = 0.0
    band_force_z = band_ratio * weight

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- settle phase --------------------------------------------------
        for _ in range(300):
            _step(model, data, standing_qpos, pelvis_id,
                  pelvis_target_z, band_force_z,
                  band_kp_xy, band_kd_xy, kp, kd)
            mujoco.mj_step(model, data)
            step_count += 1
        viewer.sync()

        # --- run -----------------------------------------------------------
        while viewer.is_running() and _running:
            t0 = time.time()

            _step(model, data, standing_qpos, pelvis_id,
                  pelvis_target_z, band_force_z,
                  band_kp_xy, band_kd_xy, kp, kd)
            mujoco.mj_step(model, data)
            step_count += 1
            viewer.sync()

            now = time.time()
            if now - last_print > 2.0:
                snap = _snap(model, data, step_count, data.time)
                com = snap.get("com_position", [])
                zmp = snap.get("zmp", [])
                ct = len(snap.get("contact_forces", []))
                bodies = len(snap.get("body_poses", {}))
                torso = snap.get("body_poses", {}).get("torso_link", {})
                pos = torso.get("position", []) if torso else []

                parts = [f"[{step_count} t={data.time:.1f}s]"]
                if len(pos) >= 3:
                    parts.append(f"pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]")
                if len(com) >= 3:
                    parts.append(f"COM=[{com[0]:.3f},{com[1]:.3f},{com[2]:.3f}]")
                if len(zmp) >= 2:
                    parts.append(f"ZMP=[{zmp[0]:.4f},{zmp[1]:.4f}]")
                parts.append(f"ct={ct} body={bodies}")
                print("  ".join(parts), flush=True)
                last_print = now

            elapsed = time.time() - t0
            if elapsed < 0.002:
                time.sleep(0.002 - elapsed)

    print(f"\nDone. Steps: {step_count}")


# ---------------------------------------------------------------------------
# Per-step control
# ---------------------------------------------------------------------------

def _step(model, data, q_des, pelvis_id, pelvis_target_z,
          band_fz, band_kp_xy, band_kd_xy, kp, kd):
    """One control step: elastic band on pelvis + joint PD + gravity comp."""

    q = data.qpos[7:]
    qd = data.qvel[6:]

    # --- elastic band on pelvis --------------------------------------------
    pelvis_pos = data.xpos[pelvis_id]
    pelvis_vel = data.cvel[pelvis_id][3:6]  # linear velocity of pelvis COM

    # Vertical: support force with height feedback (integral-like)
    dz = pelvis_target_z - pelvis_pos[2]
    fz = band_fz + 800.0 * dz - 150.0 * pelvis_vel[2]
    fz = max(0.0, fz)  # band can only pull up, not push down

    # Sagittal (X): PD centering
    fx = -band_kp_xy * pelvis_pos[0] - band_kd_xy * pelvis_vel[0]
    # Lateral (Y): PD centering
    fy = -band_kp_xy * pelvis_pos[1] - band_kd_xy * pelvis_vel[1]

    data.xfrc_applied[pelvis_id, :3] = [fx, fy, fz]
    data.xfrc_applied[pelvis_id, 3:] = 0.0

    # --- joint control: PD + live gravity compensation ---------------------
    g_comp = data.qfrc_bias[6:].copy()   # gravity + passive forces at joints

    ctrl = g_comp + kp * (q_des - q) - kd * qd

    # Mild ankle COM feedback for sagittal balance
    com = data.subtree_com[0] if data.subtree_com.shape[0] > 0 else np.zeros(3)
    com_err_x = com[0]  # COM should be at x=0
    ctrl[4] -= 60.0 * com_err_x   # left_ankle
    ctrl[9] -= 60.0 * com_err_x   # right_ankle

    data.ctrl[:] = ctrl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lower_to_ground(model, data):
    """Adjust pelvis Z so the lowest foot sole just touches z=0."""
    mujoco.mj_forward(model, data)
    lowest = 999.0
    for foot_name in ("left_ankle_link", "right_ankle_link"):
        bid = model.body(foot_name).id
        # foot collision geom extends ~0.058 m below ankle origin
        sole_z = data.xpos[bid][2] - 0.058
        if sole_z < lowest:
            lowest = sole_z
    data.qpos[2] -= lowest
    mujoco.mj_forward(model, data)


def _compute_total_mass(model) -> float:
    """Sum all body masses (excluding 'world')."""
    total = 0.0
    for bid in range(1, model.nbody):  # skip world (index 0)
        total += model.body_mass[bid]
    return total


def _joint_state(data, model):
    jpos, jvel = {}, {}
    for jid in range(model.njnt):
        name = _jname(model, jid)
        if not name:
            continue
        qa = model.jnt_qposadr[jid]
        qva = model.jnt_dofadr[jid]
        jt = model.jnt_type[jid]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            continue
        jpos[name] = float(data.qpos[qa])
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            jvel[name] = float(data.qvel[qva])
    return jpos, jvel


def _jname(model, jid):
    s = model.name_jntadr[jid]
    return model.names[s:].decode("utf-8").split("\x00")[0]


def _bname(model, bid):
    s = model.name_bodyadr[bid]
    return model.names[s:].decode("utf-8").split("\x00")[0]


def _actuator_names(model):
    names = []
    for i in range(model.nu):
        s = model.name_actuatoradr[i]
        names.append(model.names[s:].decode("utf-8").split("\x00")[0])
    return names


def _snap(model, data, step, t):
    from roboclaw.sim.mujoco_env import _compute_zmp
    jpos, jvel = _joint_state(data, model)
    com = data.subtree_com[0].copy() if data.subtree_com.shape[0] > 0 else np.zeros(3)
    zmp = _compute_zmp(model, data)
    patterns = ("torso", "pelvis", "foot", "ankle", "toe", "hand",
                "wrist", "elbow", "shoulder", "hip", "knee")
    body_poses = {}
    for bid in range(model.nbody):
        name = _bname(model, bid)
        if name and name != "world" and any(p in name for p in patterns):
            body_poses[name] = {
                "position": data.xpos[bid].tolist(),
                "rotation": data.xmat[bid].reshape(3, 3).tolist(),
            }
    contacts = []
    for i in range(data.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        contacts.append({"magnitude": float(np.linalg.norm(f[:3]))})
    return {
        "time": t, "step_count": step,
        "joint_positions": jpos, "joint_velocities": jvel,
        "com_position": com.tolist(), "zmp": list(zmp),
        "body_poses": body_poses, "contact_forces": contacts,
    }


if __name__ == "__main__":
    main()
