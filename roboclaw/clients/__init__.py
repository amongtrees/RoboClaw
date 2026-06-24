"""Model client package — remote API clients for VLN, VLA, and World Model.

Each client wraps ``httpx.AsyncClient`` and implements the
``AbstractModelClient`` contract.  The ``ModelRouter`` dispatches subtasks
to the appropriate client based on ``ModelType`` classification.
"""

from __future__ import annotations

from roboclaw.clients.model_router import ModelRouter
from roboclaw.clients.vla_client import VLAClient
from roboclaw.clients.vln_client import VLNClient
from roboclaw.clients.world_model_client import WorldModelClient

__all__ = [
    "ModelRouter",
    "VLNClient",
    "VLAClient",
    "WorldModelClient",
]
