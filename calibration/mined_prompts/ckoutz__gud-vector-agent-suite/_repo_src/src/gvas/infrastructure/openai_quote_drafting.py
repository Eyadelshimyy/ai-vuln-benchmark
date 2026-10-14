"""OpenAI reader for free-text quote requests.

The model receives the owner's text plus the matched appointment (when there is
one) and answers through a JSON schema with line items, an optional note and
the ambiguities it noticed. It is told to copy prices verbatim and never to
invent one, but nothing here trusts that: the composite drafter in
``quote_drafting`` checks every returned price against the owner's text before
a proposal exists. Failures raise ``FreeTextQuoteDraftingError`` with a fixed
message; the key and the provider's response never leave this module.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gvas.config import OpenAISettings
from gvas.domain.enums import BillingInterval
from gvas.domain.quotes import (
    FreeTextQuoteDraft,
    FreeTextQuoteDraftingError,
    FreeTextQuoteItem,
    QuoteDraftRequest,
)
from gvas.domain.usage import UsageKind, UsageLedgerPort

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
SCHEMA_NAME: Final = "free_text_quote"
SEED: Final = 0
MAX_DESCRIPTION_CHARS: Final = 200
MAX_NOTE_CHARS: Final = 400
MAX_AMBIGUITY_CHARS: Final = 200
MAX_AMBIGUITIES: Final = 3

SYSTEM_PROMPT: Final = """You turn a small-business owner's informal quote request into line items.

The owner typed the request over chat or SMS, e.g. "inspection 250" or
"2 air samples at 125 each plus the report 200, note we'll be there tuesday".
You may also receive the customer's appointment: event name, time, address,
customer name and what the customer wrote when booking.

Rules:
- Every unit price must be an amount the owner literally wrote in the request
  text, copied as written (digits only, optional decimals). NEVER invent,
  estimate, total, split or infer a price. If an item has no price in the text,
  still list the item but leave unit_price as an empty string.
- quantity is how many units of the item are billed, and only when the owner
  wrote that count for the item ("2 air samples"); default 1. Copy the owner's
  exact words that state it, number first, into quantity_text ("2 air
  samples"); empty string when the quantity is 1. A number that describes the
  job ("2 bedrooms", "3 hours away", "1500 sqft") is NOT a quantity.
- Use the appointment only to word descriptions and the note (e.g. the
  customer's booking answer "attic mold, 2 bedrooms" makes the description
  "Mold inspection - attic and 2 bedrooms"). It never contributes a price or
  a quantity.
- owner_note holds anything the owner wrote for the customer that is not an
  item (scheduling remarks, conditions); empty string when there is none.
- ambiguities lists, briefly, anything you were unsure about (an amount that
  could be a total rather than a unit price, an item without a price, ...).
- billing is "month" or "year" ONLY when the owner literally wrote a recurrence
  word ("per month", "monthly", "/mo", "yearly", "annual"); otherwise an empty
  string. Never infer recurrence from the kind of service.
Return an empty items list when the text contains no billable item."""

RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items", "owner_note", "ambiguities", "billing"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "quantity", "quantity_text", "unit_price"],
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "quantity_text": {"type": "string"},
                    "unit_price": {"type": "string"},
                },
            },
        },
        "owner_note": {"type": "string"},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "billing": {"type": "string", "enum": ["", "month", "year"]},
    },
}


class _ReportedItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    description: str
    quantity: int
    quantity_text: str = ""
    unit_price: str


class _ReportedDraft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    items: tuple[_ReportedItem, ...] = Field(default_factory=tuple)
    owner_note: str = ""
    ambiguities: tuple[str, ...] = Field(default_factory=tuple)
    billing: str = ""


class OpenAIFreeTextQuoteDrafter:
    """Implements ``FreeTextQuoteDraftingPort`` over the chat-completions endpoint."""

    def __init__(
        self,
        settings: OpenAISettings,
        client: httpx.AsyncClient,
        usage_ledger: UsageLedgerPort | None = None,
    ) -> None:
        if not settings.is_configured:
            raise FreeTextQuoteDraftingError("openai api key is not configured")
        self._settings = settings
        self._client = client
        self._usage_ledger = usage_ledger

    async def draft(self, request: QuoteDraftRequest) -> FreeTextQuoteDraft:
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}{CHAT_COMPLETIONS_PATH}",
                json=_request_body(self._settings.review_model, request),
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("quote drafting request failed: %s", type(error).__name__)
            raise FreeTextQuoteDraftingError("openai was unreachable") from error
        draft = _draft(response)
        if self._usage_ledger is not None:
            await self._usage_ledger.record(
                request.business_id,
                UsageKind.REVIEW_TOKENS,
                _total_tokens(response),
                at=datetime.now(UTC),
            )
        return draft


def _total_tokens(response: httpx.Response) -> int:
    try:
        usage = response.json().get("usage")
    except ValueError:
        return 0
    if not isinstance(usage, dict):
        return 0
    total = 0
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def _request_body(model: str, request: QuoteDraftRequest) -> dict[str, Any]:
    return {
        "model": model,
        "seed": SEED,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(request)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": RESPONSE_SCHEMA},
        },
    }


def _user_content(request: QuoteDraftRequest) -> str:
    appointment: dict[str, Any] | None = None
    if request.appointment is not None:
        appointment = {
            "event_name": request.appointment.event_name,
            "start_time": request.appointment.start_time.isoformat(),
            "address": request.appointment.address,
            "customer_name": request.appointment.invitee_name,
            "booking_notes": list(request.appointment.notes),
        }
    return json.dumps(
        {"request_text": request.request_text, "appointment": appointment},
        ensure_ascii=False,
    )


def _draft(response: httpx.Response) -> FreeTextQuoteDraft:
    if response.status_code >= 400:
        logger.warning("quote drafting returned http %s", response.status_code)
        raise FreeTextQuoteDraftingError(f"openai returned http {response.status_code}")
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not text")
        reported = _ReportedDraft.model_validate_json(content)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as error:
        logger.warning("quote drafting returned an unreadable response: %s", type(error).__name__)
        raise FreeTextQuoteDraftingError("openai returned an unreadable quote draft") from error
    items = tuple(
        FreeTextQuoteItem(
            description=_squash(item.description)[:MAX_DESCRIPTION_CHARS],
            quantity=max(item.quantity, 1),
            quantity_text=_squash(item.quantity_text) or None,
            unit_price=item.unit_price.strip() or None,
        )
        for item in reported.items
        if _squash(item.description)
    )
    note = _squash(reported.owner_note)[:MAX_NOTE_CHARS]
    ambiguities = tuple(
        _squash(entry)[:MAX_AMBIGUITY_CHARS]
        for entry in reported.ambiguities[:MAX_AMBIGUITIES]
        if _squash(entry)
    )
    interval = (
        BillingInterval(reported.billing)
        if reported.billing in {member.value for member in BillingInterval}
        else None
    )
    return FreeTextQuoteDraft(
        line_items=items, owner_note=note or None, ambiguities=ambiguities, interval=interval
    )


def _squash(value: str) -> str:
    return " ".join(value.split())
