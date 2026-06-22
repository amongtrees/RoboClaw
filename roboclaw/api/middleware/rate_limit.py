"""Simple rate-limiting middleware (Phase 1: in-memory)."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter.

    Production should use Redis-backed sliding window.
    """

    def __init__(self, app, max_requests_per_minute: int = 60):
        super().__init__(app)
        self._max_requests = max_requests_per_minute
        self._windows: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        # Use client IP as key
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0

        # Clean old entries
        if client in self._windows:
            self._windows[client] = [t for t in self._windows[client] if t > window_start]
        else:
            self._windows[client] = []

        # Check rate
        if len(self._windows[client]) >= self._max_requests:
            return JSONResponse(
                {"error": "rate_limit_exceeded", "retry_after_s": 60},
                status_code=429,
            )

        self._windows[client].append(now)
        return await call_next(request)
