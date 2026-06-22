"""Admin routes: health, readiness, metrics."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Liveness probe — is the API process running?"""
    return {"status": "ok", "service": "roboclaw-api"}


@router.get("/health/ready")
async def readiness_check():
    """Readiness probe — are all dependencies available?

    Checks: PostgreSQL, Redis, Milvus, RabbitMQ connectivity.
    Phase 1: always returns ready (lazy initialization).
    """
    dependencies = {
        "postgres": "ok",
        "redis": "ok",
        "milvus": "ok",
        "rabbitmq": "ok",
    }
    all_ok = all(v == "ok" for v in dependencies.values())
    return {
        "status": "ready" if all_ok else "not_ready",
        "dependencies": dependencies,
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint.

    Phase 1 stub — returns basic agent metrics.
    """
    return {
        "roboclaw_agents_active": 0,
        "roboclaw_tasks_completed": 0,
        "roboclaw_tasks_failed": 0,
        "roboclaw_recovery_attempts": 0,
    }
