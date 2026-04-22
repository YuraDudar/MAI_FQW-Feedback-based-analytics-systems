import asyncio
import json
import logging
from typing import Callable, Awaitable

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

import sys
sys.path.insert(0, "/app")
from infrastructure.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_ANALYSIS_DONE,
    KAFKA_CONSUMER_GROUP_BACKEND,
    KAFKA_ACKS,
    KAFKA_ENABLE_IDEMPOTENCE,
    KAFKA_MAX_IN_FLIGHT,
    KAFKA_RETRIES,
)

logger = logging.getLogger(__name__)


class KafkaProducerClient:
    def __init__(self):
        self._producer: AIOKafkaProducer | None = None

    async def start(self):
        self._producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            acks=KAFKA_ACKS,
            enable_idempotence=KAFKA_ENABLE_IDEMPOTENCE,
            max_in_flight_requests_per_connection=KAFKA_MAX_IN_FLIGHT,
            retries=KAFKA_RETRIES,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k else None,
        )
        await self._producer.start()
        logger.info("Kafka Producer запущен")

    async def stop(self):
        if self._producer:
            await self._producer.stop()

    async def send(self, topic: str, value: dict, key: str | None = None):
        if not self._producer:
            raise RuntimeError("Kafka Producer не инициализирован")
        try:
            await self._producer.send_and_wait(topic, value=value, key=key)
            logger.debug("Отправлено в топик %s: %s", topic, value)
        except KafkaError as exc:
            logger.error("Ошибка отправки в Kafka топик %s: %s", topic, exc)
            raise


class KafkaConsumerManager:
    def __init__(self):
        self._tasks: list[asyncio.Task] = []
        self._handlers: dict[str, Callable[[dict], Awaitable[None]]] = {}

    def register(self, topic: str, handler: Callable[[dict], Awaitable[None]]):
        self._handlers[topic] = handler

    async def start(self):
        from consumers.analysis_done import handle_analysis_done
        self.register(KAFKA_TOPIC_ANALYSIS_DONE, handle_analysis_done)

        for topic, handler in self._handlers.items():
            task = asyncio.create_task(self._consume_loop(topic, handler))
            self._tasks.append(task)
            logger.info("Запущен консьюмер для топика: %s", topic)

    async def stop(self):
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _consume_loop(self, topic: str, handler: Callable[[dict], Awaitable[None]]):
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_CONSUMER_GROUP_BACKEND,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await consumer.start()
        try:
            async for msg in consumer:
                try:
                    await handler(msg.value)
                    await consumer.commit()
                except Exception as exc:
                    logger.error("Ошибка обработки сообщения из %s: %s", topic, exc)
        except asyncio.CancelledError:
            pass
        finally:
            await consumer.stop()


kafka_producer = KafkaProducerClient()
kafka_consumer_manager = KafkaConsumerManager()


async def get_kafka_producer() -> KafkaProducerClient:
    return kafka_producer
