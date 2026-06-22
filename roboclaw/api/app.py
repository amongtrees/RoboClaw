"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from roboclaw.api.middleware.logging import LoggingMiddleware
from roboclaw.api.middleware.rate_limit import RateLimitMiddleware
from roboclaw.api.routes import admin, agent, a2a_routes, episode, task, telemetry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    logger.info("RoboClaw API starting up...")
    # Initialize connections (Redis, DB, Milvus, RabbitMQ)
    # These are lazily initialized on first request in Phase 1
    yield
    logger.info("RoboClaw API shutting down...")
    # Close connections


def create_app(settings: Any = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional Settings object. If None, uses defaults.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="RoboClaw API",
        description="Production-grade vertical agent framework for humanoid robotics",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    origins = settings.api.cors_origins if settings else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=60)

    # Register routes
    app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])
    app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])
    app.include_router(task.router, prefix="/api/v1", tags=["Task"])
    app.include_router(episode.router, prefix="/api/v1", tags=["Episode"])
    app.include_router(telemetry.router, prefix="/api/v1", tags=["Telemetry"])
    app.include_router(a2a_routes.router, prefix="", tags=["A2A"])

    # Store settings in app state
    app.state.settings = settings

    return app
