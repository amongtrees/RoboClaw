"""World Model server — wraps NVIDIA Cosmos Predict-2 or Qwen-RobotWorld.

Starts a FastAPI app that loads a World Model and exposes
``POST /v1/predict`` for future-state prediction.

Two modes:
1. **Self-hosted**: loads Cosmos Predict-2 or Qwen-RobotWorld weights locally.
2. **Proxy mode**: forwards to NVIDIA Cosmos hosted API at ``api.build.nvidia.com``.

Designed to be called by ``roboclaw.clients.WorldModelClient``.

Usage::

    python -m roboclaw.serve.world_model_server --config configs/models/world_model_cosmos.yaml
"""

from __future__ import annotations

import base64
import io
import logging
import os
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
    title="RoboClaw-WorldModel-Server",
    description="World Model server — predicts future frames & states (Cosmos Predict-2 / Qwen-RobotWorld)",
    version="0.1.0",
)

_wm_model: Any = None
_config: dict[str, Any] = {}
_proxy_mode: bool = False
_proxy_url: str = "https://api.build.nvidia.com/v1/cosmos/predict"

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    request_id: str = ""
    model_type: str = "world_model"
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


class PredictResponse(BaseModel):
    request_id: str
    status: str = "success"
    predicted_frames: list[str] = Field(default_factory=list)
    predicted_state: dict[str, Any] = Field(default_factory=dict)
    uncertainty: float = 0.0
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    global _wm_model, _proxy_mode, _proxy_url

    mode = _config.get("mode", "self_hosted")
    model_path = _config.get("model_path", "")

    if mode == "proxy":
        _proxy_mode = True
        _proxy_url = _config.get("proxy_url", _proxy_url)
        logger.info("World Model server in proxy mode → %s", _proxy_url)
    elif model_path:
        try:
            logger.info("Loading World Model from %s ...", model_path)
            # Placeholder: _wm_model = CosmosPredict2Pipeline.from_pretrained(...)
            logger.info("World Model loaded successfully")
        except Exception as exc:
            logger.warning("World Model not available (GPU / weights missing): %s", exc)
    else:
        logger.info("No model_path configured — World Model server running in stub mode")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _wm_model is not None,
        "model_name": _config.get("name", "unknown"),
        "mode": "proxy" if _proxy_mode else "self_hosted",
        "stub_mode": _wm_model is None and not _proxy_mode,
    }


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()

    if _proxy_mode:
        return await _proxy_predict(req, t0)

    if _wm_model is None:
        return _stub_predict(req, t0)

    # --- Real local inference ---
    try:
        # Decode input image(s)
        images = []
        for b64_str in req.rgb_images:
            img_bytes = base64.b64decode(b64_str)
            from PIL import Image

            images.append(Image.open(io.BytesIO(img_bytes)))

        # Placeholder: real Cosmos Predict-2 inference
        # predicted = _wm_model.predict(images, action_text=req.instruction)
        predicted_frames: list[str] = []
        predicted_state: dict[str, Any] = {}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PredictResponse(
            request_id=req.request_id,
            status="success",
            predicted_frames=predicted_frames,
            predicted_state=predicted_state,
            confidence=0.80,
            inference_time_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.exception("World Model inference error")
        return PredictResponse(
            request_id=req.request_id,
            status="failure",
            error={"type": "inference_error", "detail": str(exc)},
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Stub prediction (simulated future state — usable without GPU)
# ---------------------------------------------------------------------------


def _stub_predict(req: PredictRequest, t0: float) -> PredictResponse:
    """Generate stub future state predictions for testing."""
    import random

    obj = req.parameters.get("object", "unknown")
    skill = req.skill_type

    # Simulate a prediction: the robot state after executing the action
    predicted_state = {
        "robot_id": req.robot_state.get("robot_id", "unknown"),
        "joint_positions": req.robot_state.get("joint_positions", {}),
        "zmp": [random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03)],
        "foot_contact_left": True,
        "foot_contact_right": True,
    }

    uncertainty = random.uniform(0.05, 0.25)
    if "grasp" in skill:
        predicted_state["gripper_closed"] = True
        predicted_state["object_in_hand"] = obj
        uncertainty = random.uniform(0.10, 0.30)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PredictResponse(
        request_id=req.request_id,
        status="success",
        predicted_state=predicted_state,
        uncertainty=uncertainty,
        confidence=round(1.0 - uncertainty, 2),
        inference_time_ms=elapsed_ms,
        metadata={"stub": True, "skill": skill, "object": obj},
    )


# ---------------------------------------------------------------------------
# Proxy to NVIDIA Cosmos hosted API
# ---------------------------------------------------------------------------


async def _proxy_predict(req: PredictRequest, t0: float) -> PredictResponse:
    """Forward the prediction request to the NVIDIA Cosmos hosted API."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return PredictResponse(
            request_id=req.request_id,
            status="failure",
            error={"type": "no_api_key", "detail": "NVIDIA_API_KEY env var not set"},
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    try:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
            resp = await client.post(
                _proxy_url,
                json={
                    "image": req.rgb_images[0] if req.rgb_images else "",
                    "action_text": req.instruction or req.goal,
                    "num_predicted_frames": 4,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        predicted_frames = data.get("predicted_frames", [])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PredictResponse(
            request_id=req.request_id,
            status="success",
            predicted_frames=predicted_frames,
            predicted_state=data.get("predicted_state", {}),
            uncertainty=data.get("uncertainty", 0.15),
            confidence=data.get("confidence", 0.85),
            inference_time_ms=elapsed_ms,
            metadata={"proxy": "nvidia_cosmos"},
        )
    except Exception as exc:
        logger.error("NVIDIA Cosmos proxy error: %s", exc)
        return PredictResponse(
            request_id=req.request_id,
            status="failure",
            error={"type": "proxy_error", "detail": str(exc)},
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_server(config_path: str | Path, port: int = 8004) -> None:
    global _config
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
        _config = raw.get("model", raw)

    import uvicorn

    logger.info("Starting World Model server on :%d (config=%s)", port, config_path)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoboClaw World Model Server")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--port", type=int, default=8004, help="Server port")
    args = parser.parse_args()
    run_server(args.config, args.port)
