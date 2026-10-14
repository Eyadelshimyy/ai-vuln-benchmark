"""Calendly webhook HTTP endpoint.

The route verifies the shared-secret signature, translates the payload and
hands the booking event to the application service; only signature and parse
failures change the status code — every handled event is acknowledged so
Calendly stops redelivering.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gvas.infrastructure.calendly.events import CalendlyPayloadError
from gvas.infrastructure.calendly.ingress import CalendlyWebhookIngress
from gvas.infrastructure.calendly.signature import (
    SIGNATURE_HEADER,
    CalendlySignatureError,
)


def create_calendly_router(
    ingress: CalendlyWebhookIngress, *, path: str = "/calendly/events"
) -> APIRouter:
    router = APIRouter()

    @router.post(path)
    async def calendly_events(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            outcome = await ingress.handle(
                body=body, signature=request.headers.get(SIGNATURE_HEADER)
            )
        except CalendlySignatureError:
            return JSONResponse(status_code=401, content={"error": "invalid_signature"})
        except CalendlyPayloadError:
            return JSONResponse(status_code=400, content={"error": "invalid_payload"})
        return JSONResponse(status_code=200, content={"status": outcome.result.value})

    return router
