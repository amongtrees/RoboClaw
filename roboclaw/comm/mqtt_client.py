"""MQTT client stub — lightweight sensor/telemetry pub-sub.

Phase 1 stub. In production, used for low-bandwidth telemetry
from embedded sensors and edge devices.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MQTTClient:
    """Lightweight MQTT client for sensor telemetry.

    Phase 1 stub.
    """

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883) -> None:
        self._host = broker_host
        self._port = broker_port
        self._connected = False

    async def connect(self) -> None:
        logger.info(f"MQTT client in stub mode (would connect to {self._host}:{self._port})")

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        logger.debug(f"[STUB] MQTT publish to {topic}: {payload}")

    async def subscribe(self, topic: str, callback: Any) -> None:
        logger.debug(f"[STUB] MQTT subscribe to {topic}")

    async def close(self) -> None:
        self._connected = False
