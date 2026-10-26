"""SNS event publishing for domain events."""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

from app.config import settings

logger = structlog.get_logger()


class _UUIDEncoder(json.JSONEncoder):
    def default(self, o: object) -> Any:
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class EventPublisher:
    """Publishes domain events to AWS SNS."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):  # noqa: ANN202
        if self._client is None:
            import boto3

            kwargs = {"region_name": settings.aws_region}
            if settings.aws_endpoint_url:
                kwargs["endpoint_url"] = settings.aws_endpoint_url
            self._client = boto3.client("sns", **kwargs)
        return self._client

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        if not settings.sns_enabled:
            logger.info("sns_event_skipped", event_type=event_type)
            return

        message = {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": payload,
        }

        try:
            client = self._get_client()
            await asyncio.to_thread(
                client.publish,
                TopicArn=settings.sns_topic_arn,
                Message=json.dumps(message, cls=_UUIDEncoder),
                MessageAttributes={
                    "event_type": {"DataType": "String", "StringValue": event_type}
                },
            )
            logger.info("sns_event_published", event_type=event_type)
        except Exception:
            logger.exception("sns_publish_failed", event_type=event_type)


event_publisher = EventPublisher()
