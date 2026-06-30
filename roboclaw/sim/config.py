"""Simulation configuration models.

These models define the sim-related settings that plug into
``RobotConfig`` and the per-embodiment YAML files.

All sim settings are optional and degrade gracefully when
MuJoCo is not installed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MujocoConfig(BaseModel):
    """MuJoCo engine-level configuration.

    These map directly to MuJoCo's ``mjOption`` fields and control
    how the physics solver behaves.
    """

    gravity: list[float] = Field(
        default=[0.0, 0.0, -9.81],
        description="Gravity vector [x, y, z] in m/s².",
    )
    solver: str = Field(
        default="Newton",
        description="Constraint solver algorithm: Newton | CG | PGS.",
    )
    iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum solver iterations per step.",
    )
    tolerance: float = Field(
        default=1e-8,
        gt=0,
        description="Solver convergence tolerance.",
    )
    integrator: str = Field(
        default="implicitfast",
        description="Numerical integrator: Euler | RK4 | implicit | implicitfast.",
    )
    cone: str = Field(
        default="pyramidal",
        description="Friction cone model: pyramidal | elliptic.",
    )
    jacobian: str = Field(
        default="auto",
        description="Jacobian computation: auto | dense | sparse.",
    )


class SimConfig(BaseModel):
    """Top-level simulation configuration.

    Lives under ``robot.sim`` in the YAML config and controls
    whether simulation is active and which backend to use.
    """

    enabled: bool = Field(
        default=False,
        description="Enable physics simulation (False = use asyncio.sleep stubs).",
    )
    backend: str = Field(
        default="mujoco",
        description="Simulation backend: mujoco | none. Extensible to isaac, pybullet.",
    )
    model_path: str = Field(
        default="",
        description="Path to MJCF or URDF model file for the robot.",
    )
    render: bool = Field(
        default=False,
        description="Open a MuJoCo viewer window (requires display).",
    )
    headless: bool = Field(
        default=True,
        description="Run in headless mode (no viewer, no GPU rendering).",
    )
    timestep: float = Field(
        default=0.002,
        gt=0,
        description="Physics timestep in seconds (MuJoCo default 0.002).",
    )
    control_decimation: int = Field(
        default=5,
        ge=1,
        description="Control update every N physics steps (5 → 100 Hz control at 0.002 ts).",
    )
    sensor_rate_hz: int = Field(
        default=100,
        ge=1,
        description="Sensor snapshot rate in Hz.",
    )
    initial_qpos: str = Field(
        default="",
        description="Optional path to initial joint positions file.",
    )

    @property
    def control_timestep(self) -> float:
        """Effective control timestep = timestep * decimation."""
        return self.timestep * self.control_decimation

    @property
    def control_rate_hz(self) -> float:
        """Effective control rate in Hz."""
        return 1.0 / self.control_timestep
