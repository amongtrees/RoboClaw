"""Authentication middleware stub.

In production, this would validate JWT tokens, API keys, or OAuth2.
Phase 1: pass-through (no auth).
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware stub.

    Production implementation validates credentials and injects
    user/role information into request.state.
    """

    async def dispatch(self, request: Request, call_next):
        # Phase 1: pass-through
        return await call_next(request)
