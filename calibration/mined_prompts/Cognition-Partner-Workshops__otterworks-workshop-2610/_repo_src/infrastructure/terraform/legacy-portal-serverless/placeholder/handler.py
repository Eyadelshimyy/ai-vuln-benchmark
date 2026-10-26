import json
import os


def handler(event, context):
    return {
        "statusCode": 501,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "error": "not implemented",
            "context": os.environ.get("CONTEXT"),
            "path": event.get("rawPath"),
        }),
    }
