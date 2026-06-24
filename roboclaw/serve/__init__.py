"""Model-serving sub-package — standalone FastAPI microservices.

Each server wraps a foundation model (VLN / VLA / World Model) behind
a standard REST API so that the RoboClaw agent can call it over HTTP.

Usage (standalone, one per model)::

    python -m roboclaw.serve.vln_server --config configs/models/vln_qwen_robotnav.yaml
    python -m roboclaw.serve.vla_server --config configs/models/vla_groot_n1.yaml
    python -m roboclaw.serve.world_model_server --config configs/models/world_model_cosmos.yaml

All servers expose:
* ``GET  /health``       — liveness check
* ``POST /v1/<action>``  — inference endpoint (navigate / act / predict)
"""

from __future__ import annotations
