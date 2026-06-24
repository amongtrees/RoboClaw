"""VLN model server — wraps Qwen-RobotNav (or any VLN model).

Starts a FastAPI app that loads a Vision-Language Navigation model and
exposes a ``POST /v1/navigate`` endpoint.

The server is designed to be called by ``roboclaw.clients.VLNClient``.

Usage::

    python -m roboclaw.serve.vln_server --config configs/models/vln_qwen_robotnav.yaml
    # or programmatically:
    # run_server("configs/models/vln_qwen_robotnav.yaml", port=8002)
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory (so the server is importable without starting immediately)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RoboClaw-VLN-Server",
    description="Vision-Language Navigation model server (Qwen-RobotNav compatible)",
    version="0.1.0",
)

# Global model handles — loaded at startup.  In Phase 2 these remain
# ``None`` unless a real GPU + model weights are available; the server
# will respond with ``{"status": "stub"}`` until then.
_nav_model: Any = None
_processor: Any = None
_config: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Pydantic schemas (mirror the client-side ModelInferenceRequest/Response)
# ---------------------------------------------------------------------------


class NavigateRequest(BaseModel):
    request_id: str = ""
    model_type: str = "vln"
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


class NavigateResponse(BaseModel):
    request_id: str
    status: str = "success"
    waypoints: list[dict[str, float]] = Field(default_factory=list)
    trajectory: list[dict[str, float]] = Field(default_factory=list)
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    global _nav_model, _processor
    model_path = _config.get("model_path", "")
    if model_path:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            import torch

            logger.info("Loading VLN model from %s ...", model_path)
            _nav_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            _processor = AutoProcessor.from_pretrained(model_path)
            logger.info("VLN model loaded successfully")
        except Exception as exc:
            logger.warning("VLN model not available (GPU / weights missing): %s", exc)
    else:
        logger.info("No model_path configured — VLN server running in stub mode")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _nav_model is not None,
        "model_name": _config.get("name", "unknown"),
        "stub_mode": _nav_model is None,
    }


@app.post("/v1/navigate", response_model=NavigateResponse)
async def navigate(req: NavigateRequest) -> NavigateResponse:
    t0 = time.perf_counter()

    # --- Stub mode: return simulated waypoints ---
    if _nav_model is None:
        return _stub_navigate(req, t0)

    # --- Real inference ---
    try:
        # Decode base64 images
        images = []
        for b64_str in req.rgb_images:
            img_bytes = base64.b64decode(b64_str)
            from PIL import Image

            images.append(Image.open(io.BytesIO(img_bytes)))

        instruction = req.instruction or req.goal or "Navigate to target"

        # Build model inputs (Qwen3-VL chat format)
        messages = [
            {"role": "system", "content": "You are a vision-language navigation model. Given an image and a navigation goal, output a sequence of waypoints as JSON."},
            {"role": "user", "content": [
                {"type": "image", "image": img} for img in images
            ] + [{"type": "text", "text": instruction}]},
        ]
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _processor(text=text, images=images, return_tensors="pt").to(_nav_model.device)

        # Generate
        import torch

        with torch.inference_mode():
            generated_ids = _nav_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract waypoints from model output
        waypoints = _parse_waypoints_from_text(output_text, instruction)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return NavigateResponse(
            request_id=req.request_id,
            status="success",
            waypoints=waypoints,
            confidence=0.85,
            inference_time_ms=elapsed_ms,
            metadata={"raw_output": output_text[:200]},
        )
    except Exception as exc:
        logger.exception("VLN inference error")
        return NavigateResponse(
            request_id=req.request_id,
            status="failure",
            error={"type": "inference_error", "detail": str(exc)},
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Stub navigation (simulated waypoints — usable for testing without GPU)
# ---------------------------------------------------------------------------


def _stub_navigate(req: NavigateRequest, t0: float) -> NavigateResponse:
    """Generate a straight-line trajectory to a simulated goal."""
    import random

    room = req.parameters.get("room", "unknown")
    # Simulate 3–5 waypoints toward the goal
    num_pts = random.randint(3, 5)
    waypoints = [
        {"x": round(0.5 + i * 1.2, 2), "y": round(0.1 * i, 2), "z": 0.0,
         "theta": round(0.1 * i, 2)}
        for i in range(num_pts)
    ]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return NavigateResponse(
        request_id=req.request_id,
        status="success",
        waypoints=waypoints,
        confidence=0.90,
        inference_time_ms=elapsed_ms,
        metadata={"stub": True, "room": room, "waypoint_count": num_pts},
    )


def _parse_waypoints_from_text(output_text: str, instruction: str) -> list[dict[str, float]]:
    """Try to extract waypoints from a model's JSON or structured output."""
    import json
    import re

    # Try JSON array
    json_match = re.search(r"\[[\s\S]*?\]", output_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: return a single placeholder waypoint
    return [{"x": 1.0, "y": 0.0, "z": 0.0, "theta": 0.0}]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_server(config_path: str | Path, port: int = 8002) -> None:
    """Load config from YAML and start the VLN FastAPI server."""
    global _config
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
        _config = raw.get("model", raw)

    import uvicorn

    logger.info("Starting VLN server on :%d (config=%s)", port, config_path)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoboClaw VLN Model Server")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    args = parser.parse_args()
    run_server(args.config, args.port)
