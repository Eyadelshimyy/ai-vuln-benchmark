from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gvas.infrastructure.telnyx.events import TelnyxPayloadError
from gvas.infrastructure.telnyx.ingress import TelnyxMessagingIngress
from gvas.infrastructure.telnyx.signature import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    TelnyxSignatureError,
)


def create_telnyx_router(
    ingress: TelnyxMessagingIngress, *, path: str = "/telnyx/messaging"
) -> APIRouter:
    """Telnyx messaging webhook endpoint: verify, persist, acknowledge."""

    router = APIRouter()

    @router.post(path)
    async def telnyx_messaging(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            outcome = await ingress.handle(
                body=body,
                signature=request.headers.get(SIGNATURE_HEADER),
                timestamp=request.headers.get(TIMESTAMP_HEADER),
            )
        except TelnyxSignatureError:
            return JSONResponse({"status": "invalid_signature"}, status_code=401)
        except TelnyxPayloadError:
            return JSONResponse({"status": "invalid_payload"}, status_code=400)
        return JSONResponse({"status": outcome.result.value}, status_code=200)

    return router
