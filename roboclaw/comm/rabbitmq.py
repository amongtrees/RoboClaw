"""RabbitMQ async producer/consumer for inter-service messaging.

Used for: sensor event bus, task event notifications, A2A messaging.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RabbitMQClient:
    """Async RabbitMQ client for pub/sub messaging.

    Phase 1 stub — logs instead of actually connecting to RabbitMQ.
    Production: uses aio-pika for async AMQP.
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._connected = False
        self._handlers: dict[str, list[Any]] = {}

    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        if self._config:
            try:
                import aio_pika

                self._connection = await aio_pika.connect_robust(self._config.url)
                self._channel = await self._connection.channel()
                self._connected = True
                logger.info("Connected to RabbitMQ")
                return
            except ImportError:
                logger.warning("aio-pika not installed — RabbitMQ unavailable")
            except Exception as e:
                logger.warning(f"RabbitMQ connection failed: {e}")

        logger.info("RabbitMQ client in stub mode")

    async def publish(self, exchange_name: str, routing_key: str, message: dict[str, Any]) -> None:
        """Publish a message to an exchange."""
        if self._connected:
            try:
                exchange = await self._channel.get_exchange(exchange_name)
                await exchange.publish(
                    aio_pika.Message(body=json.dumps(message).encode()),
                    routing_key=routing_key,
                )
                return
            except Exception as e:
                logger.error(f"RabbitMQ publish failed: {e}")

        # Stub: log the message
        logger.debug(f"[STUB] Published to {exchange_name}/{routing_key}: {message}")

    async def subscribe(self, exchange_name: str, routing_key: str, handler: Any) -> None:
        """Subscribe to messages from an exchange."""
        if self._connected:
            try:
                exchange = await self._channel.get_exchange(exchange_name)
                queue = await self._channel.declare_queue(exclusive=True)
                await queue.bind(exchange, routing_key=routing_key)
                await queue.consume(handler)
                logger.info(f"Subscribed to {exchange_name}/{routing_key}")
                return
            except Exception as e:
                logger.error(f"RabbitMQ subscribe failed: {e}")

        # Stub: register handler
        key = f"{exchange_name}:{routing_key}"
        self._handlers.setdefault(key, []).append(handler)
        logger.debug(f"[STUB] Subscribed to {key}")

    async def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self._connected:
            try:
                await self._channel.close()
                await self._connection.close()
            except Exception:
                pass
        self._connected = False
