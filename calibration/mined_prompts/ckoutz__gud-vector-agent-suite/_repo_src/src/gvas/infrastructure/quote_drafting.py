"""Deterministic quote drafting.

GVAS structures the owner's own prices; it never estimates one. The owner
therefore states every price explicitly in a small line-oriented format, and
this parser either produces exactly what was written or refuses with an
owner-facing message.

    quote:
    customer: person@example.com
    currency: USD
    item: 2 | Air sampling | 125.00
    item: 1 | Report | 200.00
    note: on-site visit scheduled for Tuesday
    billing: monthly

``customer``, ``currency`` and at least one ``item`` are required; ``billing``
is optional (``monthly`` or ``yearly``) and makes the quote recur; ``customer``
may instead arrive as ``request.recipient`` when the workflow resolved it from an
appointment, and an explicit line still wins. ``for`` is accepted and ignored
here (the workflow consumes it). Keys and
surrounding whitespace are case- and space-insensitive; amounts are not. Amounts
are read as exact minor units, so no floating point value is ever involved.

The pilot prices in USD only. A currency is not a formatting detail: its
minor-unit exponent decides what ``125`` means, so accepting any three-letter
code while assuming two decimals would misprice JPY by a factor of a hundred.
Anything outside :data:`gvas.domain.money.MINOR_UNIT_DIGITS` is refused.

:class:`ModelAssistedQuoteDrafter` keeps that parser as the first pass and, only
when it refuses, lets a :class:`~gvas.domain.quotes.FreeTextQuoteDraftingPort`
read the text (``quote: inspection 250``). The model proposes items; this module
still decides the prices: every unit price it returns must be an amount the
owner literally wrote, otherwise nothing is drafted and the owner is asked.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from gvas.domain.enums import (
    BillingInterval,
    HostedLinkKind,
    QuoteBilling,
    RecipientAddressKind,
)
from gvas.domain.messages import CustomerRecipient
from gvas.domain.money import (
    UnsupportedCurrencyError,
    minor_unit_digits,
    minor_unit_scale,
    supported_currencies,
)
from gvas.domain.ports import QuoteDraftingPort
from gvas.domain.quotes import (
    CUSTOMER_LINE_KEYS,
    CUSTOMER_NAME_LINE_KEY,
    FreeTextQuoteDraft,
    FreeTextQuoteDraftingError,
    FreeTextQuoteDraftingPort,
    HostedLinkReference,
    QuoteDraftProposal,
    QuoteDraftRejectedError,
    QuoteDraftRequest,
    QuoteLineItem,
)
from gvas.domain.usage import UsageCeilingGuard, UsageKind
from gvas.infrastructure.hosted_links import PORTAL_LOGIN_LINK_REFERENCE

logger = logging.getLogger(__name__)

#: Every accepted amount is written in this currency; see ``_parse_currency``.
QUOTED_CURRENCY: Final = "USD"
QUOTED_CURRENCY_DIGITS: Final = minor_unit_digits(QUOTED_CURRENCY)
QUOTED_CURRENCY_SCALE: Final = minor_unit_scale(QUOTED_CURRENCY)
AMOUNT_PATTERN: Final = re.compile(rf"^\d{{1,12}}(\.\d{{1,{QUOTED_CURRENCY_DIGITS}}})?$")
EMAIL_PATTERN: Final = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
CURRENCY_PATTERN: Final = re.compile(r"^[A-Za-z]{3}$")

FORMAT_HELP: Final = (
    "Use:\nquote:\ncustomer: person@example.com\ncurrency: USD\nitem: 2 | Air sampling | 125.00"
    "\n(customer: is optional when a booking matches.)"
)
FREE_TEXT_UNAVAILABLE: Final = f"The quote could not be drafted from your message. {FORMAT_HELP}"
FREE_TEXT_CEILING_REACHED: Final = (
    f"The monthly limit for drafting quotes from free text is reached. {FORMAT_HELP}"
)
STRUCTURED_LINE_KEYS: Final = frozenset(
    {*CUSTOMER_LINE_KEYS, CUSTOMER_NAME_LINE_KEY, "billing", "currency", "item", "note", "quote"}
)
BILLING_VALUES: Final[dict[str, BillingInterval]] = {
    "monthly": BillingInterval.MONTH,
    "month": BillingInterval.MONTH,
    "yearly": BillingInterval.YEAR,
    "annual": BillingInterval.YEAR,
    "annually": BillingInterval.YEAR,
    "year": BillingInterval.YEAR,
}
#: The only wording that makes a free-text quote recur; the model may confirm
#: an interval but never introduce one the owner did not write.
RECURRENCE_PATTERNS: Final[tuple[tuple[re.Pattern[str], BillingInterval], ...]] = (
    (
        re.compile(r"\b(per month|monthly|a month|each month)\b|/\s?mo\b", re.I),
        BillingInterval.MONTH,
    ),
    (
        re.compile(r"\b(per year|yearly|annual(?:ly)?|a year|each year)\b|/\s?yr\b", re.I),
        BillingInterval.YEAR,
    ),
)
#: Amounts as an owner writes them: ``250``, ``250.00``, ``$250``, ``1,250``.
WRITTEN_AMOUNT_PATTERN: Final = re.compile(
    rf"\$?\s?(\d{{1,3}}(?:,\d{{3}})+|\d{{1,12}})(?:\.(\d{{1,{QUOTED_CURRENCY_DIGITS}}}))?(?!\d)"
)


def parse_amount_minor(value: str) -> int:
    """Read an exact amount in minor units; prices are never inferred or rounded."""

    text = value.strip()
    if not AMOUNT_PATTERN.match(text):
        raise QuoteDraftRejectedError(
            f"'{text}' is not a valid amount. Write amounts like 125 or 125.00."
        )
    whole, _, fraction = text.partition(".")
    return int(whole) * QUOTED_CURRENCY_SCALE + int(
        fraction.ljust(QUOTED_CURRENCY_DIGITS, "0") or 0
    )


def _parse_item(value: str) -> QuoteLineItem:
    fields = [field.strip() for field in value.split("|")]
    if len(fields) != 3:
        raise QuoteDraftRejectedError(
            "Each item needs 'item: <quantity> | <description> | <unit price>'."
        )
    quantity_text, description, price_text = fields
    if not quantity_text.isdigit() or int(quantity_text) < 1:
        raise QuoteDraftRejectedError(
            f"'{quantity_text}' is not a valid quantity. Use a whole number of 1 or more."
        )
    if not description:
        raise QuoteDraftRejectedError("Each item needs a description.")
    return QuoteLineItem(
        description=description,
        quantity=int(quantity_text),
        unit_price_minor=parse_amount_minor(price_text),
    )


class DeterministicQuoteDrafter:
    """Formats the owner's explicit prices into a proposal, or refuses."""

    def __init__(self, portal_link_reference: str = PORTAL_LOGIN_LINK_REFERENCE) -> None:
        self._portal_link_reference = portal_link_reference

    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal:
        recipient: CustomerRecipient | None = request.recipient
        currency: str | None = None
        note: str | None = None
        interval: BillingInterval | None = None
        line_items: list[QuoteLineItem] = []
        for line in request.request_text.splitlines():
            entry = line.strip()
            if not entry:
                continue
            key, separator, value = entry.partition(":")
            if not separator:
                raise QuoteDraftRejectedError(
                    f"'{entry}' is not a recognised quote line. {FORMAT_HELP}"
                )
            field = key.strip().casefold()
            content = value.strip()
            if field in CUSTOMER_LINE_KEYS:
                recipient = CustomerRecipient(
                    address=_parse_email(content), address_kind=RecipientAddressKind.EMAIL
                )
            elif field == CUSTOMER_NAME_LINE_KEY:
                continue
            elif field == "currency":
                currency = _parse_currency(content)
            elif field == "item":
                line_items.append(_parse_item(content))
            elif field == "note":
                note = content or None
            elif field == "billing":
                interval = _parse_billing(content)
            elif field == "quote":
                continue
            else:
                raise QuoteDraftRejectedError(
                    f"'{key.strip()}' is not a supported quote field. {FORMAT_HELP}"
                )
        missing = [
            name
            for name, present in (
                ("customer", recipient is not None),
                ("currency", currency is not None),
                ("item", bool(line_items)),
            )
            if not present
        ]
        if missing or recipient is None or currency is None:
            raise QuoteDraftRejectedError(
                f"The quote is missing: {', '.join(missing)}. {FORMAT_HELP}"
            )
        return QuoteDraftProposal(
            quote_id=request.quote_id,
            business_id=request.business_id,
            recipient=recipient,
            currency=currency,
            line_items=tuple(line_items),
            owner_note=note,
            hosted_links=(
                HostedLinkReference(
                    kind=HostedLinkKind.SIGNUP, reference=self._portal_link_reference
                ),
            ),
            billing=QuoteBilling.RECURRING if interval is not None else QuoteBilling.ONE_TIME,
            interval=interval,
        )


def is_structured_request(request_text: str) -> bool:
    """True when every line is a ``key: value`` line of the structured format."""

    lines = [line.strip() for line in request_text.splitlines() if line.strip()]
    return bool(lines) and all(
        separator and key.strip().casefold() in STRUCTURED_LINE_KEYS
        for key, separator, _ in (line.partition(":") for line in lines)
    )


def written_amounts_minor(text: str) -> frozenset[int]:
    """Every amount that appears literally in ``text``, in minor units."""

    found: set[int] = set()
    for match in WRITTEN_AMOUNT_PATTERN.finditer(text):
        whole, fraction = match.group(1).replace(",", ""), match.group(2) or ""
        found.add(
            int(whole) * QUOTED_CURRENCY_SCALE + int(fraction.ljust(QUOTED_CURRENCY_DIGITS, "0"))
        )
    return frozenset(found)


WORD_PATTERN: Final = re.compile(r"[A-Za-z][A-Za-z-]*")
COUNTED_ATTRIBUTE_PATTERN: Final = re.compile(r"\d[\d.,]*\s*(?:x|×)?\s*[A-Za-z][A-Za-z-]*")
WORD_STEM_LENGTH: Final = 4


def quantity_is_written(
    text: str, quantity: int, description: str, quantity_text: str | None
) -> bool:
    """True when the owner wrote ``quantity_text`` (the drafter's source for the
    quantity, ``"2 air samples"``) and it really counts this item: it appears
    in ``text`` with the number standing alone (not inside ``$1,250`` or
    ``125.00``), and every word after the number starts like a word of the
    description that is not itself counted there. ``"2 bedrooms"`` for
    ``Inspection for 2 bedrooms`` describes the job, not how many are billed,
    and ``"2 mold inspections"`` does not count ``Mold remediation``."""

    return quantity_source_occurrences(text, quantity, description, quantity_text) > 0


def quantity_source_occurrences(
    text: str, quantity: int, description: str, quantity_text: str | None
) -> int:
    """How many times ``quantity_text`` occurs in ``text`` as a count of this
    item (see ``quantity_is_written``); 0 when it is not one."""

    if quantity_text is None:
        return 0
    span = " ".join(quantity_text.split())
    match = re.fullmatch(
        rf"{quantity}(?![\d.,])\s*(?:x|×)?\s*({WORD_PATTERN.pattern}(?:\s+{WORD_PATTERN.pattern})*)",
        span,
        re.IGNORECASE,
    )
    if match is None:
        return 0
    uncounted = COUNTED_ATTRIBUTE_PATTERN.sub(" ", description)
    stems = {
        word.casefold()[:WORD_STEM_LENGTH]
        for word in WORD_PATTERN.findall(uncounted)
        if len(word) >= 3
    }
    words = [
        word.casefold()[:WORD_STEM_LENGTH]
        for word in WORD_PATTERN.findall(match.group(1))
        if len(word) >= 3
    ]
    if not words or any(word not in stems for word in words):
        return 0
    haystack = " ".join(text.split()).casefold()
    needle = span.casefold()
    occurrences = 0
    start = haystack.find(needle)
    while start != -1:
        if start == 0 or haystack[start - 1] not in "0123456789.,$":
            occurrences += 1
        start = haystack.find(needle, start + 1)
    return occurrences


def written_amount_minor(value: str) -> int | None:
    """``$1,250.00`` -> 125000; ``None`` when ``value`` is not a single amount."""

    match = WRITTEN_AMOUNT_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    whole, fraction = match.group(1).replace(",", ""), match.group(2) or ""
    return int(whole) * QUOTED_CURRENCY_SCALE + int(fraction.ljust(QUOTED_CURRENCY_DIGITS, "0"))


def missing_price_question(descriptions: list[str]) -> str:
    quoted = ", ".join(f"'{description}'" for description in descriptions)
    noun = "price" if len(descriptions) == 1 else "prices"
    return f"What's the {noun} for {quoted}? Send the quote again with every price included."


class ModelAssistedQuoteDrafter:
    """Structured format first; free text through a model only when it refuses.

    The model never sets a price. Each unit price it returns is looked up among
    the amounts written in the owner's text; a price that is not there, or an
    item without one, turns into one question naming the items and nothing is
    drafted. The customer comes from an explicit ``customer:`` line or from the
    matched appointment, exactly as for the structured format.
    """

    def __init__(
        self,
        structured: QuoteDraftingPort,
        free_text: FreeTextQuoteDraftingPort,
        *,
        ceilings: UsageCeilingGuard | None = None,
        now: Callable[[], datetime] | None = None,
        portal_link_reference: str = PORTAL_LOGIN_LINK_REFERENCE,
    ) -> None:
        self._structured = structured
        self._free_text = free_text
        self._ceilings = ceilings or UsageCeilingGuard()
        self._now = now or (lambda: datetime.now(UTC))
        self._portal_link_reference = portal_link_reference

    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal:
        try:
            return await self._structured.draft(request)
        except QuoteDraftRejectedError:
            # A request written entirely in the structured format has a
            # structured mistake; the parser's message says which, and a model
            # would only guess at the owner's intent.
            if is_structured_request(request.request_text):
                raise
        recipient, free_text = _split_customer_lines(request)
        if await self._ceilings.is_reached(
            request.business_id, UsageKind.REVIEW_TOKENS, now=self._now()
        ):
            raise QuoteDraftRejectedError(FREE_TEXT_CEILING_REACHED)
        try:
            draft = await self._free_text.draft(
                request.model_copy(update={"request_text": free_text})
            )
        except FreeTextQuoteDraftingError as error:
            logger.warning("free-text quote drafting failed for %s: %s", request.quote_id, error)
            raise QuoteDraftRejectedError(FREE_TEXT_UNAVAILABLE) from error
        line_items = _priced_line_items(free_text, draft)
        interval = written_recurrence(free_text)
        if draft.interval is not None and interval is None:
            logger.info("free-text draft proposed recurrence the owner did not write; ignored")
        return QuoteDraftProposal(
            quote_id=request.quote_id,
            business_id=request.business_id,
            recipient=recipient,
            currency=QUOTED_CURRENCY,
            line_items=line_items,
            owner_note=draft.owner_note,
            hosted_links=(
                HostedLinkReference(
                    kind=HostedLinkKind.SIGNUP, reference=self._portal_link_reference
                ),
            ),
            risk_flags=draft.ambiguities,
            drafted_from_free_text=True,
            billing=QuoteBilling.RECURRING if interval is not None else QuoteBilling.ONE_TIME,
            interval=interval,
        )


def written_recurrence(text: str) -> BillingInterval | None:
    """The recurrence the owner literally wrote, or None for a one-time quote.

    Like prices, recurrence is never inferred: the model's ``interval`` is
    ignored unless the owner's own words name that interval.
    """

    found = {interval for pattern, interval in RECURRENCE_PATTERNS if pattern.search(text)}
    if len(found) > 1:
        raise QuoteDraftRejectedError(
            "Your message mentions both monthly and yearly billing. Send the quote again"
            " with one billing interval."
        )
    return next(iter(found), None)


def _split_customer_lines(request: QuoteDraftRequest) -> tuple[CustomerRecipient, str]:
    """The recipient, and the text without ``customer:``/``for:`` lines.

    Only that remainder goes to the model, so a customer's address or name is
    never read as an item and digits in an e-mail address never pass as a price.
    """

    recipient = request.recipient
    remainder: list[str] = []
    for line in request.request_text.splitlines():
        key, separator, value = line.partition(":")
        field = key.strip().casefold()
        if separator and field in CUSTOMER_LINE_KEYS:
            recipient = CustomerRecipient(
                address=_parse_email(value.strip()), address_kind=RecipientAddressKind.EMAIL
            )
        elif separator and field in {CUSTOMER_NAME_LINE_KEY, "quote"}:
            continue
        else:
            remainder.append(line)
    if recipient is None:
        raise QuoteDraftRejectedError(f"The quote is missing: customer. {FORMAT_HELP}")
    free_text = "\n".join(remainder).strip()
    if not free_text:
        raise QuoteDraftRejectedError(f"The quote is missing: item. {FORMAT_HELP}")
    return recipient, free_text


def _priced_line_items(request_text: str, draft: FreeTextQuoteDraft) -> tuple[QuoteLineItem, ...]:
    if not draft.line_items:
        raise QuoteDraftRejectedError(f"No priced items were found in your message. {FORMAT_HELP}")
    written = written_amounts_minor(request_text)
    line_items: list[QuoteLineItem] = []
    unpriced: list[str] = []
    used_quantity_texts: Counter[str] = Counter()
    for item in draft.line_items:
        minor = None if item.unit_price is None else written_amount_minor(item.unit_price)
        if minor is None or minor not in written:
            unpriced.append(item.description)
            continue
        quantity_text = " ".join((item.quantity_text or "").split()).casefold()
        # One written count serves one item: "2 inspections" cannot count both
        # "Mold inspection" and "Radon inspection" unless it was written twice.
        if item.quantity > 1 and used_quantity_texts[quantity_text] >= quantity_source_occurrences(
            request_text, item.quantity, item.description, item.quantity_text
        ):
            raise QuoteDraftRejectedError(
                f"Your message does not say {item.quantity} × '{item.description}'."
                " Send the quote again with the quantity written out."
            )
        if item.quantity > 1:
            used_quantity_texts[quantity_text] += 1
        line_items.append(
            QuoteLineItem(
                description=item.description, quantity=item.quantity, unit_price_minor=minor
            )
        )
    if unpriced:
        raise QuoteDraftRejectedError(missing_price_question(unpriced))
    return tuple(line_items)


def _parse_email(value: str) -> str:
    if not EMAIL_PATTERN.match(value):
        raise QuoteDraftRejectedError(
            f"'{value}' is not a valid customer email address."
            if value
            else "The quote needs a customer email address."
        )
    return value


def _parse_billing(value: str) -> BillingInterval | None:
    key = value.strip().casefold()
    if key in {"", "once", "one_time", "one-time", "one time"}:
        return None
    interval = BILLING_VALUES.get(key)
    if interval is None:
        raise QuoteDraftRejectedError(
            f"'{value}' is not a valid billing value. Use 'billing: monthly' or 'billing: yearly'."
        )
    return interval


def _parse_currency(value: str) -> str:
    if not CURRENCY_PATTERN.match(value):
        raise QuoteDraftRejectedError(
            f"'{value}' is not a valid currency. Use a three-letter code such as USD."
        )
    code = value.upper()
    try:
        minor_unit_digits(code)
    except UnsupportedCurrencyError as error:
        raise QuoteDraftRejectedError(
            f"Quotes cannot be priced in {code}. "
            f"Supported currencies: {', '.join(supported_currencies())}."
        ) from error
    return code
