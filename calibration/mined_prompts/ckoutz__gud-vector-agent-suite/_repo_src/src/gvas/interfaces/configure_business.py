"""Hosted-quote configuration for one business.

Sets the public site origin the hosted quote links point at, the
customer-facing display name, the booking link the contact form shows, the
publishable key the site's frontend calls the public API with, and the
connected-account id reserved for future per-business payouts. Re-running with
different values updates the row; a value that is not supplied is left alone.
When no ``--public-key`` is given and the business has none, a fresh one is
generated; it is printed so it can be copied into the site's frontend config.

The ``--intake-*`` options set the website booking agent's profile: a brief
describing the business and what the agent books, the questions to ask beyond
name/email/phone, and the first message a visitor reads.

``--owner-email`` names who signs in to the owner dashboard: that address
gets an owner link from the same portal login page customers use.

    gvas-configure-business --business-id <uuid> --site-url https://gudvector.com \
        --display-name "Güd Vector" --calendly-url https://calendly.com/gudvector \
        --intake-brief "..." --intake-questions "..." --intake-opening "..."
"""

import argparse
import asyncio
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID

from gvas.config import Settings
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import (
    INTAKE_BRIEF_MAX_CHARS,
    INTAKE_OFFER_LINE_MAX_CHARS,
    INTAKE_OPENING_MAX_CHARS,
    INTAKE_QUESTIONS_MAX_CHARS,
    INTAKE_VISIT_QUESTION_MAX_CHARS,
)
from gvas.domain.invoices import INVOICE_PREFIX_MAX_CHARS
from gvas.domain.reporting import normalize_email_address
from gvas.domain.repositories import (
    BusinessRecord,
    is_local_host,
    normalize_site_url,
)
from gvas.domain.time_zones import normalize_time_zone
from gvas.infrastructure.db import create_engine, create_session_factory
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory

PUBLIC_KEY_PREFIX = "gvb_"

#: A public key travels as a single URL path segment, so only unreserved
#: characters are allowed; the column caps it at 255 characters.
PUBLIC_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,254}")


class ConfigureBusinessInputError(ValueError):
    """The supplied configuration is missing or malformed."""


@dataclass(frozen=True)
class ConfigureBusinessRequest:
    business_id: BusinessId
    site_url: str | None
    display_name: str | None
    calendly_url: str | None
    stripe_account_id: str | None
    public_key: str | None
    intake_brief: str | None = None
    intake_questions: str | None = None
    intake_opening: str | None = None
    intake_offer_line: str | None = None
    intake_visit_question: str | None = None
    invoice_prefix: str | None = None
    notification_email: str | None = None
    owner_email: str | None = None
    timezone: str | None = None


def build_request(arguments: argparse.Namespace) -> ConfigureBusinessRequest:
    raw_id = (arguments.business_id or "").strip()
    if not raw_id:
        raise ConfigureBusinessInputError("--business-id is required")
    try:
        business_id = BusinessId(UUID(raw_id))
    except ValueError as error:
        raise ConfigureBusinessInputError("--business-id must be a UUID") from error

    site_url = _optional(arguments.site_url)
    if site_url is not None:
        try:
            site_url = normalize_site_url(site_url)
        except ValueError as error:
            raise ConfigureBusinessInputError(f"--site-url: {error}") from error
    calendly_url = _optional(arguments.calendly_url)
    if calendly_url is not None:
        # Not an origin: booking links legitimately carry a path, so the
        # check only demands an absolute http(s) URL with a real host.
        try:
            parts = urlsplit(calendly_url)
            host = parts.hostname
            parts.port  # noqa: B018 - property access raises on a malformed port
        except ValueError as error:
            raise ConfigureBusinessInputError("--calendly-url is not a parseable URL") from error
        if parts.scheme.lower() not in ("http", "https") or not host:
            raise ConfigureBusinessInputError("--calendly-url must be an absolute http(s) URL")
        if parts.scheme.lower() == "http" and not is_local_host(host):
            # Customer-facing links must not send people over cleartext.
            raise ConfigureBusinessInputError(
                "--calendly-url must use https outside local development"
            )
    intake_brief = _optional_text(arguments.intake_brief, "--intake-brief", INTAKE_BRIEF_MAX_CHARS)
    intake_questions = _optional_text(
        arguments.intake_questions, "--intake-questions", INTAKE_QUESTIONS_MAX_CHARS
    )
    intake_opening = _optional_text(
        arguments.intake_opening, "--intake-opening", INTAKE_OPENING_MAX_CHARS
    )
    intake_offer_line = _optional_text(
        getattr(arguments, "intake_offer_line", None),
        "--intake-offer-line",
        INTAKE_OFFER_LINE_MAX_CHARS,
    )
    intake_visit_question = _optional_text(
        getattr(arguments, "intake_visit_question", None),
        "--intake-visit-question",
        INTAKE_VISIT_QUESTION_MAX_CHARS,
    )
    raw_invoice_prefix = getattr(arguments, "invoice_prefix", None)
    # An empty --invoice-prefix "" switches invoices off.
    invoice_prefix = (
        ""
        if raw_invoice_prefix is not None and not raw_invoice_prefix.strip()
        else _optional_text(raw_invoice_prefix, "--invoice-prefix", INVOICE_PREFIX_MAX_CHARS)
    )
    notification_email = _optional(arguments.notification_email)
    if getattr(arguments, "clear_notification_email", False):
        if notification_email is not None:
            raise ConfigureBusinessInputError(
                "--clear-notification-email cannot be combined with --notification-email"
            )
        notification_email = ""
    elif notification_email is not None:
        normalized = normalize_email_address(notification_email)
        if normalized is None:
            raise ConfigureBusinessInputError("--notification-email must be an e-mail address")
        notification_email = normalized
    owner_email = _optional(getattr(arguments, "owner_email", None))
    if owner_email is not None:
        normalized_owner = normalize_email_address(owner_email)
        if normalized_owner is None:
            raise ConfigureBusinessInputError("--owner-email must be an e-mail address")
        owner_email = normalized_owner
    timezone = _optional(getattr(arguments, "timezone", None))
    if timezone is not None:
        try:
            timezone = normalize_time_zone(timezone)
        except ValueError as error:
            raise ConfigureBusinessInputError(f"--timezone {error}") from error
    if all(
        value is None
        for value in (
            site_url,
            timezone,
            notification_email,
            owner_email,
            _optional(arguments.display_name),
            calendly_url,
            _optional(arguments.stripe_account_id),
            _optional(arguments.public_key),
            intake_brief,
            intake_questions,
            intake_opening,
            intake_offer_line,
            intake_visit_question,
            invoice_prefix,
        )
    ):
        raise ConfigureBusinessInputError("nothing to configure; pass at least one option")
    public_key = _optional(arguments.public_key)
    if public_key is not None and PUBLIC_KEY_PATTERN.fullmatch(public_key) is None:
        raise ConfigureBusinessInputError(
            "--public-key must be a single URL-safe path segment (letters, digits and . _ ~ -)"
        )
    return ConfigureBusinessRequest(
        business_id=business_id,
        site_url=site_url,
        display_name=_optional(arguments.display_name),
        calendly_url=calendly_url,
        stripe_account_id=_optional(arguments.stripe_account_id),
        public_key=public_key,
        intake_brief=intake_brief,
        intake_questions=intake_questions,
        intake_opening=intake_opening,
        intake_offer_line=intake_offer_line,
        intake_visit_question=intake_visit_question,
        invoice_prefix=invoice_prefix,
        notification_email=notification_email,
        owner_email=owner_email,
        timezone=timezone,
    )


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_text(value: str | None, flag: str, max_chars: int) -> str | None:
    text = _optional(value)
    if text is None:
        return None
    text = " ".join(text.split())
    if len(text) > max_chars:
        raise ConfigureBusinessInputError(f"{flag} must be at most {max_chars} characters")
    return text


async def run_configure(request: ConfigureBusinessRequest) -> BusinessRecord:
    engine = create_engine(Settings().database_url)
    try:
        async with SqlUnitOfWorkFactory(create_session_factory(engine))() as unit_of_work:
            existing = await unit_of_work.businesses.get(request.business_id)
            if existing is None:
                raise ConfigureBusinessInputError(
                    f"business {request.business_id} does not exist; run gvas-bootstrap first"
                )
            public_key = request.public_key
            if public_key is None and existing.public_key is None:
                public_key = PUBLIC_KEY_PREFIX + secrets.token_urlsafe(16)
            business = await unit_of_work.businesses.configure_site(
                request.business_id,
                site_url=request.site_url,
                display_name=request.display_name,
                calendly_url=request.calendly_url,
                stripe_account_id=request.stripe_account_id,
                public_key=public_key,
                intake_brief=request.intake_brief,
                intake_questions=request.intake_questions,
                intake_opening=request.intake_opening,
                intake_offer_line=request.intake_offer_line,
                intake_visit_question=request.intake_visit_question,
                invoice_prefix=request.invoice_prefix,
                notification_email=request.notification_email,
                owner_email=request.owner_email,
                timezone=request.timezone,
                now=datetime.now(UTC),
            )
            await unit_of_work.commit()
            return business
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure a business's hosted-quote site settings"
    )
    parser.add_argument("--business-id", required=True)
    parser.add_argument("--site-url")
    parser.add_argument("--display-name")
    parser.add_argument("--calendly-url")
    parser.add_argument("--stripe-account-id")
    parser.add_argument("--public-key")
    parser.add_argument(
        "--intake-brief", help="1-3 sentences: what the business does and what the agent books"
    )
    parser.add_argument(
        "--intake-questions", help="what the agent finds out beyond name, email and phone"
    )
    parser.add_argument("--intake-opening", help="the agent's first message to a visitor")
    parser.add_argument(
        "--invoice-prefix",
        help='turn invoices on, numbered PREFIX-1001 and up (e.g. "LL"); "" turns them off',
    )
    parser.add_argument("--intake-offer-line", help="the agent's line above the offered times")
    parser.add_argument(
        "--intake-visit-question",
        help="asked once a time is picked; the answer is saved with the request",
    )
    parser.add_argument(
        "--clear-notification-email",
        action="store_true",
        help="stop e-mailing copies of website notices",
    )
    parser.add_argument(
        "--notification-email",
        help="owner inbox that gets a copy of every website booking/escalation/payment notice",
    )
    parser.add_argument(
        "--owner-email",
        help="the address that signs in to the owner dashboard through the portal login",
    )
    parser.add_argument(
        "--timezone", help="IANA zone the business works in, e.g. America/Los_Angeles"
    )
    try:
        request = build_request(parser.parse_args(argv))
        business = asyncio.run(run_configure(request))
    except ConfigureBusinessInputError as error:
        parser.exit(2, f"error: {error}\n")
    print(  # noqa: T201
        f"business {business.business_id} site_url {business.site_url} "
        f"display_name {business.display_name} calendly_url {business.calendly_url} "
        f"public_key {business.public_key} "
        f"intake_profile {'custom' if business.intake_profile.is_configured else 'default'} "
        f"notification_email {business.notification_email} "
        f"owner_email {business.owner_email} "
        f"timezone {business.timezone}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
