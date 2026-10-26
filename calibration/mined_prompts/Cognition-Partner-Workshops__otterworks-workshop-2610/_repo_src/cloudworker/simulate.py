#!/usr/bin/env python3
"""Publish file_shared events to the cloud-worker SNS topic.

The message body matches FileEvent in services/file-service/src/events.rs, so
notification-service parses it exactly as it parses a real share. Owner and
recipient are seeded users from scripts/seed.py. Each published event prints
one JSON line with its messageId, fileId and recipient.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / ".state" / "outputs.json"

OWNER_ID = "5eed0001-0000-4000-a000-000000000001"
RECIPIENT_ID = "5eed0002-0000-4000-a000-000000000002"
DEMO_SOURCE = "aws-cloud-worker"


def topic_from_outputs() -> str:
    if not OUTPUTS.exists():
        return ""
    return str(json.loads(OUTPUTS.read_text()).get("sns_topic_arn", ""))


def file_shared(owner: str, recipient: str) -> dict[str, object]:
    return {
        "eventType": "file_shared",
        "fileId": str(uuid.uuid4()),
        "ownerId": owner,
        "folderId": None,
        "sharedWithUserId": recipient,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def publish_command(topic: str, region: str, body: str) -> list[str]:
    attributes = json.dumps({"demoSource": {"DataType": "String", "StringValue": DEMO_SOURCE}})
    return [
        "aws", "sns", "publish",
        "--region", region,
        "--topic-arn", topic,
        "--message", body,
        "--message-attributes", attributes,
        "--query", "MessageId",
        "--output", "text",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--topic-arn", default=os.environ.get("CW_SNS_TOPIC_ARN") or topic_from_outputs())
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--owner", default=OWNER_ID)
    parser.add_argument("--recipient", default=RECIPIENT_ID)
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("CW_DRY_RUN") == "1")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if not args.topic_arn:
        parser.error("no topic: pass --topic-arn, set CW_SNS_TOPIC_ARN, or run cw.sh up")

    for _ in range(args.count):
        event = file_shared(args.owner, args.recipient)
        body = json.dumps(event, separators=(",", ":"))
        cmd = publish_command(args.topic_arn, args.region, body)
        if args.dry_run:
            print("+ " + shlex.join(cmd), file=sys.stderr)
            message_id = "dry-run"
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
            if result.returncode != 0:
                print(result.stderr.strip(), file=sys.stderr)
                return result.returncode
            message_id = result.stdout.strip()
        print(json.dumps({"messageId": message_id, "fileId": event["fileId"], "userId": args.recipient}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
