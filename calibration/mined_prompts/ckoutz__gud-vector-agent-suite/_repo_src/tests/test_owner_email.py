"""Owner notice e-mail rendering."""

import pytest

from gvas.domain.owner_email import (
    OwnerEmailAction,
    OwnerEmailContent,
    OwnerEmailDetail,
    owner_notice_content,
    render_owner_email_html,
    render_owner_email_text,
)


def sample_content() -> OwnerEmailContent:
    return OwnerEmailContent(
        heading="New booking request #65e7b537",
        intro="Jane <Doe> asked for a visit.",
        details=(
            OwnerEmailDetail(label="Customer", value="Jane <Doe> & Co"),
            OwnerEmailDetail(label="Requested time", value="Tue Oct 6, 9:00 AM (America/Denver)"),
        ),
        actions=(
            OwnerEmailAction(label="Approve", url="https://gvas.example/a?t=1&x=2", primary=True),
            OwnerEmailAction(label="Decline", url="https://gvas.example/d"),
        ),
        commands=("Or reply `approve booking 65e7b537` / `decline booking 65e7b537 <reason>`.",),
        business_name="Güd Vector",
    )


def test_html_is_self_contained_escaped_and_has_both_buttons() -> None:
    html = render_owner_email_html(sample_content())
    assert "max-width:34rem" in html
    assert "<img" not in html.lower()
    assert "<link" not in html.lower() and "<style" not in html.lower()
    assert "-apple-system" in html
    assert "Jane &lt;Doe&gt; &amp; Co" in html
    assert "<Doe>" not in html
    assert 'href="https://gvas.example/a?t=1&amp;x=2"' in html
    assert ">Approve</a>" in html and ">Decline</a>" in html
    assert "approve booking 65e7b537" in html


def test_text_part_keeps_details_links_and_commands() -> None:
    text = render_owner_email_text(sample_content())
    assert text.startswith("New booking request #65e7b537")
    assert "Customer: Jane <Doe> & Co" in text
    assert "Approve: https://gvas.example/a?t=1&x=2" in text
    assert "`approve booking 65e7b537`" in text


def test_action_urls_must_be_web_links() -> None:
    with pytest.raises(ValueError):
        OwnerEmailAction(label="Approve", url="javascript:alert(1)")


def test_channel_notice_text_becomes_structured_content() -> None:
    content = owner_notice_content(
        "Customer needs a person — Jane Doe asked for a human.\n"
        "Phone: +15555550100\n"
        "Reply `approve booking abcd1234` to act."
    )
    assert content.heading == "Customer needs a person"
    assert content.intro == "Jane Doe asked for a human."
    assert content.details == (OwnerEmailDetail(label="Phone", value="+15555550100"),)
    assert content.commands
