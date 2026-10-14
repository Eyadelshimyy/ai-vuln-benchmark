from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gvas.infrastructure.slack.events import SlackPayloadError
from gvas.infrastructure.slack.ingress import SlackEventIngress, SlackIngressResult
from gvas.infrastructure.slack.signature import SlackSignatureError

SIGNATURE_HEADER = "X-Slack-Signature"
TIMESTAMP_HEADER = "X-Slack-Request-Timestamp"
RETRY_NUM_HEADER = "X-Slack-Retry-Num"
RETRY_REASON_HEADER = "X-Slack-Retry-Reason"


def create_slack_router(ingress: SlackEventIngress, *, path: str = "/slack/events") -> APIRouter:
    """Slack Request URL endpoint: verify, persist, acknowledge."""

    router = APIRouter()

    @router.post(path)
    async def slack_events(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            outcome = await ingress.handle(
                body=body,
                signature=request.headers.get(SIGNATURE_HEADER),
                timestamp=request.headers.get(TIMESTAMP_HEADER),
                retry_num=request.headers.get(RETRY_NUM_HEADER),
                retry_reason=request.headers.get(RETRY_REASON_HEADER),
            )
        except SlackSignatureError:
            return JSONResponse({"status": "invalid_signature"}, status_code=401)
        except SlackPayloadError:
            return JSONResponse({"status": "invalid_payload"}, status_code=400)
        if outcome.result is SlackIngressResult.CHALLENGE:
            return JSONResponse({"challenge": outcome.challenge}, status_code=200)
        return JSONResponse({"status": outcome.result.value}, status_code=200)

    return router
