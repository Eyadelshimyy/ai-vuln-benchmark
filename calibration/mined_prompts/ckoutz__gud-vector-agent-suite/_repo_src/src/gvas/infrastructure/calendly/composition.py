"""Calendly webhook composition: bind the verified endpoint to the service."""

from fastapi import APIRouter

from gvas.application.intake import IntakeBookingEventService
from gvas.infrastructure.calendly.config import (
    CalendlySettings,
    parse_calendly_installations,
)
from gvas.infrastructure.calendly.ingress import CalendlyWebhookIngress
from gvas.interfaces.http.calendly import create_calendly_router


def build_calendly_webhook_router(
    service: IntakeBookingEventService, settings: CalendlySettings | None = None
) -> APIRouter:
    """HTTP route that verifies Calendly's signature and applies booking events.

    Mounted only when ``webhook_signing_key`` is set; the subscription itself
    is created out-of-band (see docs/deployment.md).
    """

    resolved = settings or CalendlySettings()
    ingress = CalendlyWebhookIngress(
        service,
        parse_calendly_installations(resolved.installations),
        signing_key=resolved.webhook_signing_key,
    )
    return create_calendly_router(ingress)
