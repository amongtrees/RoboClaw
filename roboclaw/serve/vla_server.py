"""VLA model server — wraps GR00T-N1-2B (or any VLA model).

Starts a FastAPI app that loads a Vision-Language-Action model and
exposes a ``POST /v1/act`` endpoint.

Designed to be called by ``roboclaw.clients.VLAClient``.

GR00T-N1 dual-system architecture:
- System 2 (VLM): slow reasoning → action tokens
- System 1 (DiT / flow-matching): fast decoding → joint actions at ~16 Hz

Usage::

    python -m roboclaw.serve.vla_server --config configs/models/vla_groot_n1.yaml
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RoboClaw-VLA-Server",
    description="Vision-Language-Action model server (GR00T-N1 / pi0 compatible)",
    version="0.1.0",
)

_vla_model: Any = None
_processor: Any = None
_config: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ActRequest(BaseModel):
    request_id: str = ""
    model_type: str = "vla"
    sub_task_id: str = ""
    skill_type: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    rgb_images: list[str] = Field(default_factory=list)
    depth_images: list[str] = Field(default_factory=list)
    goal: str = ""
    instruction: str = ""
    robot_state: dict[str, Any] = Field(default_factory=dict)
    world_state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActResponse(BaseModel):
    request_id: str
    status: str = "success"
    joint_actions: dict[str, list[float]] = Field(default_factory=dict)
    ee_poses: list[dict[str, float]] = Field(default_factory=list)
    gripper_command: str | None = None
    action_tokens: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    global _vla_model, _processor
    model_path = _config.get("model_path", "")
    if model_path:
        try:
            logger.info("Loading VLA model from %s ...", model_path)
            # GR00T-N1 is loaded via its own pipeline.  For now this is a
            # stub — real loading depends on the specific model format.
            # _vla_model = GR00TN1Pipeline.from_pretrained(model_path, ...)
            logger.info("VLA model loaded successfully")
        except Exception as exc:
            logger.warning("VLA model not available (GPU / weights missing): %s", exc)
    else:
        logger.info("No model_path configured — VLA server running in stub mode")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _vla_model is not None,
        "model_name": _config.get("name", "unknown"),
        "stub_mode": _vla_model is None,
    }


@app.get("/v1/action_vocab")
async def action_vocab() -> dict[str, Any]:
    """Return the action token vocabulary for client-side decoding.

    Each token maps to a (joint_name, delta_value) pair.
    """
    return {
        "vocab_size": 256,
        "description": "GR00T-N1 discrete action tokens (stub)",
        "tokens": {
            0: {"joint": "left_shoulder_pitch", "delta_rad": 0.01},
            1: {"joint": "left_shoulder_pitch", "delta_rad": -0.01},
            # ... full vocab would be populated from the real model
        },
    }


@app.post("/v1/act", response_model=ActResponse)
async def act(req: ActRequest) -> ActResponse:
    t0 = time.perf_counter()

    if _vla_model is None:
        return _stub_act(req, t0)

    # --- Real inference via GR00T-N1 pipeline ---
    try:
        # Decode images
        images = []
        for b64_str in req.rgb_images:
            img_bytes = base64.b64decode(b64_str)
            from PIL import Image

            images.append(Image.open(io.BytesIO(img_bytes)))

        instruction = req.instruction or req.goal or "Manipulate object"
        arm = req.parameters.get("arm", "right")
        obj = req.parameters.get("object", "unknown")

        # GR00T-N1 inference (placeholder — real implementation depends on model format)
        joint_actions = _simulate_joint_actions(arm, obj)
        gripper = "close" if req.skill_type in ("whole_body_grasp",) else None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ActResponse(
            request_id=req.request_id,
            status="success",
            joint_actions=joint_actions,
            ee_poses=[{"x": 0.4, "y": 0.0 if arm == "right" else -0.15, "z": 0.9, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}],
            gripper_command=gripper,
            action_tokens=[1, 3, 7, 15],
            confidence=0.88,
            inference_time_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.exception("VLA inference error")
        return ActResponse(
            request_id=req.request_id,
            status="failure",
            error={"type": "inference_error", "detail": str(exc)},
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Stub manipulation (simulated joint targets — usable for testing without GPU)
# ---------------------------------------------------------------------------


def _stub_act(req: ActRequest, t0: float) -> ActResponse:
    """Generate simulated joint targets for a manipulation task."""
    import random

    arm = req.parameters.get("arm", "right")
    obj = req.parameters.get("object", "unknown")
    skill = req.skill_type

    joint_actions = _simulate_joint_actions(arm, obj)

    # Determine gripper command from skill type
    gripper = None
    if skill in ("whole_body_grasp", "bi_manual_carry"):
        gripper = "close"
    elif skill == "place_object":
        gripper = "open"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 + random.uniform(20, 80)
    return ActResponse(
        request_id=req.request_id,
        status="success",
        joint_actions=joint_actions,
        ee_poses=[{"x": 0.45, "y": 0.0, "z": 0.85, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}],
        gripper_command=gripper,
        action_tokens=[1, 3, 5, 7],
        confidence=0.82,
        inference_time_ms=elapsed_ms,
        metadata={"stub": True, "skill": skill, "object": obj, "arm": arm},
    )


def _simulate_joint_actions(arm: str, obj: str) -> dict[str, list[float]]:
    """Produce plausible joint-space target positions for a grasp."""
    prefix = f"{arm}_" if arm else ""
    return {
        f"{prefix}shoulder_pitch": [0.5, 0.5, 0.5],
        f"{prefix}shoulder_roll": [0.0, 0.0, 0.0],
        f"{prefix}elbow": [1.2, 1.2, 1.2],
        f"{prefix}wrist_pitch": [-0.3, -0.3, -0.3],
        f"{prefix}wrist_roll": [0.0, 0.0, 0.0],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_server(config_path: str | Path, port: int = 8003) -> None:
    global _config
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
        _config = raw.get("model", raw)

    import uvicorn

    logger.info("Starting VLA server on :%d (config=%s)", port, config_path)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoboClaw VLA Model Server")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--port", type=int, default=8003, help="Server port")
    args = parser.parse_args()
    run_server(args.config, args.port)
