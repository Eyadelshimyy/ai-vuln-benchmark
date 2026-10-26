"""Email transport for error reports.

Per ERROR_HANDLING.md — every reportable error/job failure must be deliverable
to a configured support recipient. The send must be best-effort; a failed send
must not mask the original failure, but it MUST be recorded as ``email_sent``
on the job/error record (see ``ErtJobError.email_sent``).

In v1 we expose a tiny abstract :class:`EmailService` so the API/services don't
care which wire we end up using (SMTP vs Microsoft Graph is open — see
``docs/OPEN_QUESTIONS.md``). The default dev transport is :class:`NoopEmailService`
which never touches the network and always returns ``True`` (it logs the payload
so a developer can inspect it).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)


class EmailService(ABC):
    """Pluggable error-report transport."""

    @abstractmethod
    def send_error_report(
        self,
        *,
        subject: str,
        technical: dict[str, Any],
        recipient: str,
    ) -> bool:
        """Send the error report. Returns ``True`` on best-effort success."""


class NoopEmailService(EmailService):
    """Dev transport — logs the email, never sends; always returns True."""

    def send_error_report(
        self,
        *,
        subject: str,
        technical: dict[str, Any],
        recipient: str,
    ) -> bool:
        logger.info(
            "[noop-email] subject=%r recipient=%r technical_keys=%s",
            subject,
            recipient,
            sorted(technical.keys()),
        )
        return True


def get_email_service() -> EmailService:
    """Factory keyed on ``EMAIL_TRANSPORT`` config (CLAUDE.md rule 8 — no hardcodes)."""
    settings = get_settings()
    transport = settings.email_transport
    if transport == "noop":
        return NoopEmailService()
    if transport == "smtp":  # pragma: no cover — v1 stub
        raise NotImplementedError(
            "SMTP email transport is not implemented in v1. "
            "Use EMAIL_TRANSPORT=noop for local development."
        )
    if transport == "graph":  # pragma: no cover — v1 stub
        raise NotImplementedError(
            "Microsoft Graph email transport is not implemented in v1. "
            "Use EMAIL_TRANSPORT=noop for local development."
        )
    raise ValueError(f"Unknown EMAIL_TRANSPORT: {transport!r}")


__all__ = ["EmailService", "NoopEmailService", "get_email_service"]
