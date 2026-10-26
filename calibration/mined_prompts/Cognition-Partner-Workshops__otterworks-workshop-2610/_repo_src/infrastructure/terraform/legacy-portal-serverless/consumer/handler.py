"""Writes one DynamoDB item per announcement.published event from the run's bus."""

import json
import os

import boto3

TABLE = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])


def handler(event, _context):
    detail = event.get("detail") or {}
    item = {
        "eventId": event["id"],
        "announcementId": detail.get("id"),
        "title": detail.get("title"),
        "body": detail.get("body"),
        "published": bool(detail.get("published")),
        "createdAt": detail.get("createdAt"),
        "eventTime": event.get("time"),
        "source": event.get("source"),
        "detailType": event.get("detail-type"),
        "detail": json.dumps(detail, separators=(",", ":")),
    }
    TABLE.put_item(Item={k: v for k, v in item.items() if v is not None})
    print(json.dumps({"stored": item["eventId"], "announcementId": item["announcementId"]}))
    return {"stored": item["eventId"]}
