#!/usr/bin/env python3
"""OpenVLA-7B inference server — exposes /v1/act API for VLA model routing.

Matches the interface expected by roboclaw.clients.vla_client.VLAClient.

Usage:
    pip install transformers torch accelerate fastapi uvicorn
    python scripts/serve_vla.py --port 8001 --model models/ai/openvla-7b
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

parser = argparse.ArgumentParser(description="OpenVLA-7B Inference Server")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--model", type=str, default="models/ai/openvla-7b",
                    help="Path or HF repo id for OpenVLA-7B")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_vla")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

vla_model: Any = None
processor: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vla_model, processor
    logger.info("Loading OpenVLA-7B from %s (device=%s)...", args.model, args.device)
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    vla_model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    vla_model.eval()
    logger.info("OpenVLA-7B ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="RoboClaw VLA Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response (matches ModelInferenceRequest/Response)
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    request_id: str = ""
    model_type: str = "vla"
    sub_task_id: str = ""
    skill_type: str = ""
    parameters: dict[str, Any] = {}
    goal: str = ""
    instruction: str = ""
    robot_state: dict[str, Any] = {}
    world_state: dict[str, Any] = {}


class InferResponse(BaseModel):
    request_id: str
    status: str = "success"
    confidence: float = 0.8
    inference_time_ms: float = 0.0
    joint_actions: dict[str, list[float]] = {}
    ee_poses: list[dict[str, float]] = []
    gripper_command: str | None = None
    action_tokens: list[int] = []
    metadata: dict[str, Any] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "openvla-7b", "device": args.device}


@app.post("/v1/act", response_model=InferResponse)
async def infer(req: InferRequest):
    if vla_model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()

    # Build prompt from the instruction and goal
    instruction = req.instruction or req.goal or "perform the task"
    skill = req.skill_type or "manipulation"

    try:
        # OpenVLA expects: "In: {image_context}\nOut: {action_text}\nIn: {instruction}\nOut:"
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Tokenize
        inputs = processor(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = vla_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        action_text = processor.decode(outputs[0], skip_special_tokens=True)

        # Parse action tokens from output (OpenVLA outputs discretized actions)
        # For now, return the raw text + empty joint actions as stub
        # In production, map tokens → continuous actions via OpenVLA's action decoder
        elapsed = (time.time() - t0) * 1000

        logger.info("VLA inference: %s → %s (%.0fms)", skill, action_text[:80], elapsed)

        return InferResponse(
            request_id=req.request_id or f"vla_{int(t0)}",
            status="success",
            confidence=0.75,
            inference_time_ms=elapsed,
            joint_actions={
                "left_shoulder_pitch": [0.0],
                "left_elbow": [0.0],
                "right_shoulder_pitch": [0.0],
                "right_elbow": [0.0],
            },
            action_tokens=[],
            metadata={
                "raw_output": action_text,
                "model": "openvla-7b",
                "device": args.device,
            },
        )

    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        logger.error("VLA inference error: %s", exc)
        return InferResponse(
            request_id=req.request_id,
            status="failure",
            inference_time_ms=elapsed,
            metadata={"error": str(exc)},
        )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VLA server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
