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

    # --- Bootstrap ModelRouter for VLN / VLA / World Model dispatch ---
    settings = getattr(app.state, "settings", None)
    if settings and getattr(settings, "models", None):
        app.state.model_router = _create_model_router(settings)
        from roboclaw.orchestration.nodes import set_model_router
        set_model_router(app.state.model_router)
        logger.info("ModelRouter installed with models: %s",
                     list(app.state.model_router.registered_model_types))
    else:
        app.state.model_router = None
        logger.info("No model clients configured — agent will use simulated executors")

    yield

    # --- Shutdown ---
    logger.info("RoboClaw API shutting down...")
    if app.state.model_router is not None:
        await app.state.model_router.close()


def _create_model_router(settings: Any) -> Any:
    """Create a ModelRouter from the Settings.models configuration."""
    from roboclaw.clients.model_router import ModelRouter
    from roboclaw.clients.vla_client import VLAClient
    from roboclaw.clients.vln_client import VLNClient
    from roboclaw.clients.world_model_client import WorldModelClient
    from roboclaw.core.types import ModelType

    vln_client = None
    vla_client = None
    wm_client = None

    for key, cfg in settings.models.items():
        if not cfg.enabled:
            continue
        mt = cfg.model_type or key
        if mt == ModelType.VLN:
            vln_client = VLNClient(cfg)
            logger.info("Created VLN client: %s → %s%s", cfg.model_name, cfg.base_url, cfg.api_path)
        elif mt == ModelType.VLA:
            vla_client = VLAClient(cfg)
            logger.info("Created VLA client: %s → %s%s", cfg.model_name, cfg.base_url, cfg.api_path)
        elif mt == ModelType.WORLD_MODEL:
            wm_client = WorldModelClient(cfg)
            logger.info("Created World Model client: %s → %s%s", cfg.model_name, cfg.base_url, cfg.api_path)

    return ModelRouter(
        vln_client=vln_client,
        vla_client=vla_client,
        world_model_client=wm_client,
    )


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
