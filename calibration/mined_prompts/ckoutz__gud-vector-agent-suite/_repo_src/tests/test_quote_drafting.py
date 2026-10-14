from uuid import uuid4

import pytest

from gvas.domain.enums import RecipientAddressKind
from gvas.domain.identifiers import BusinessId, ConversationId, QuoteId
from gvas.domain.quotes import QuoteDraftRejectedError, QuoteDraftRequest
from gvas.infrastructure.hosted_links import PORTAL_LOGIN_LINK_REFERENCE
from gvas.infrastructure.quote_drafting import DeterministicQuoteDrafter, parse_amount_minor

VALID_REQUEST = "\n".join(
    (
        "quote:",
        "customer: person@example.com",
        "currency: usd",
        "item: 2 | Air sampling | 125.00",
        "item: 1 | Report | 200.5",
        "note: two day turnaround",
    )
)


def request_of(text: str) -> QuoteDraftRequest:
    return QuoteDraftRequest(
        quote_id=QuoteId(uuid4()),
        business_id=BusinessId(uuid4()),
        conversation_id=ConversationId(uuid4()),
        request_text=text,
        revision=1,
        idempotency_key="quote:1",
    )


@pytest.mark.parametrize(
    ("text", "minor"),
    [("125", 12500), ("125.00", 12500), ("0.05", 5), ("125.5", 12550), ("1234567.89", 123456789)],
)
def test_amounts_are_exact_minor_units(text: str, minor: int) -> None:
    assert parse_amount_minor(text) == minor


@pytest.mark.asyncio
async def test_draft_reads_the_documented_format_without_inferring_anything() -> None:
    proposal = await DeterministicQuoteDrafter().draft(request_of(VALID_REQUEST))

    assert proposal.recipient.address == "person@example.com"
    assert proposal.recipient.address_kind is RecipientAddressKind.EMAIL
    assert proposal.currency == "USD"
    assert [(item.quantity, item.unit_price_minor) for item in proposal.line_items] == [
        (2, 12500),
        (1, 20050),
    ]
    assert proposal.owner_note == "two day turnaround"
    assert [link.reference for link in proposal.hosted_links] == [PORTAL_LOGIN_LINK_REFERENCE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "quote:\ncurrency: USD\nitem: 1 | Report | 200.00",
        "quote:\ncustomer: person@example.com\nitem: 1 | Report | 200.00",
        "quote:\ncustomer: person@example.com\ncurrency: USD",
        "quote:\ncustomer: person@example.com\ncurrency: USD\nitem: 1 | Report | about 200",
        "quote:\ncustomer: person@example.com\ncurrency: USD\nitem: 1 | Report",
        "quote:\ncustomer: not-an-email\ncurrency: USD\nitem: 1 | Report | 200.00",
        "quote:\ncustomer: person@example.com\ncurrency: USD\nitem: 0 | Report | 200.00",
        "please quote air sampling for Dana",
    ],
)
async def test_draft_refuses_incomplete_or_ambiguous_requests(text: str) -> None:
    with pytest.raises(QuoteDraftRejectedError) as error:
        await DeterministicQuoteDrafter().draft(request_of(text))

    assert str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["JPY", "EUR", "jpy"])
async def test_draft_refuses_a_currency_it_cannot_price_truthfully(code: str) -> None:
    """JPY has no minor units, so pricing it as if it did would multiply by 100."""

    text = f"quote:\ncustomer: person@example.com\ncurrency: {code}\nitem: 1 | Report | 200"

    with pytest.raises(QuoteDraftRejectedError) as error:
        await DeterministicQuoteDrafter().draft(request_of(text))

    assert "USD" in str(error.value)


@pytest.mark.asyncio
async def test_draft_tolerates_case_and_whitespace_variation() -> None:
    text = "\n".join(
        (
            "Quote:",
            "  Customer :  person@example.com  ",
            "",
            "CURRENCY: usd",
            " Item :  3 |  Air sampling  |  125.00 ",
        )
    )

    proposal = await DeterministicQuoteDrafter().draft(request_of(text))

    assert proposal.line_items[0].description == "Air sampling"
    assert proposal.line_items[0].quantity == 3
