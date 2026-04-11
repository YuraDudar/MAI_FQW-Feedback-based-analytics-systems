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
    KAFKA_TOPIC_PARSE_JOBS,
    KAFKA_TOPIC_CLUSTER_JOBS,
    KAFKA_TOPIC_AUTO_REPLY_JOBS,
    KAFKA_CONSUMER_GROUP_ML,
    KAFKA_ACKS,
    KAFKA_ENABLE_IDEMPOTENCE,
    KAFKA_MAX_IN_FLIGHT,
    KAFKA_RETRIES,
)

logger = logging.getLogger(__name__)

_producer_instance: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            acks=KAFKA_ACKS,
            enable_idempotence=KAFKA_ENABLE_IDEMPOTENCE,
            max_in_flight_requests_per_connection=KAFKA_MAX_IN_FLIGHT,
            retries=KAFKA_RETRIES,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k else None,
        )
        await _producer_instance.start()
    return _producer_instance


async def publish(topic: str, value: dict, key: str | None = None):
    producer = await get_producer()
    try:
        await producer.send_and_wait(topic, value=value, key=key)
    except KafkaError as exc:
        logger.error("Ошибка отправки в Kafka %s: %s", topic, exc)
        raise


class KafkaConsumerManager:
    def __init__(self):
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        from consumers.parse_jobs import handle_parse_job
        from consumers.cluster_jobs import handle_cluster_job
        from consumers.auto_reply_jobs import handle_auto_reply_job

        pairs = [
            (KAFKA_TOPIC_PARSE_JOBS, handle_parse_job),
            (KAFKA_TOPIC_CLUSTER_JOBS, handle_cluster_job),
            (KAFKA_TOPIC_AUTO_REPLY_JOBS, handle_auto_reply_job),
        ]
        for topic, handler in pairs:
            task = asyncio.create_task(self._consume_loop(topic, handler))
            self._tasks.append(task)
            logger.info("Запущен ML консьюмер: %s", topic)

    async def stop(self):
        global _producer_instance
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if _producer_instance:
            await _producer_instance.stop()

    async def _consume_loop(self, topic: str, handler: Callable[[dict], Awaitable[None]]):
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_CONSUMER_GROUP_ML,
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
                    logger.error("Ошибка обработки %s: %s", topic, exc)
        except asyncio.CancelledError:
            pass
        finally:
            await consumer.stop()


kafka_consumer_manager = KafkaConsumerManager()
