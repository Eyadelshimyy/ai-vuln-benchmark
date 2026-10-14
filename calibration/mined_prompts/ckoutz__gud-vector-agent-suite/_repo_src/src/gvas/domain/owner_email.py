"""Owner notification e-mails.

Every owner notice e-mail — booking requests, escalations, portal service
requests, quote acceptances and payments — is rendered from one
``OwnerEmailContent`` into a plain-text part and a small inline-styled HTML
part, so they all read the same.
"""

import html
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.identifiers import BusinessId

# Endpoints of the retired reply-by-e-mail channel may still exist in the
# database; they must never anchor new owner notices.
OWNER_EMAIL_SOURCE_NAMESPACE = "email"
OWNER_EMAIL_SUBJECT_MAX_CHARS = 78
OWNER_CHANNEL_FOOTER = "Reply in your owner channel to act on this."
_CODE = re.compile(r"`([^`]+)`")
_DETAIL_LINE = re.compile(r"^(?P<label>[A-Z][A-Za-z ]{0,29}):\s+(?P<value>\S.*)$")


class OwnerEmailModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OwnerEmailDetail(OwnerEmailModel):
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)


class OwnerEmailAction(OwnerEmailModel):
    """A button; only absolute http(s) links are rendered."""

    label: str = Field(min_length=1)
    url: str = Field(min_length=1)
    primary: bool = False

    @field_validator("url")
    @classmethod
    def _web_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("owner e-mail actions must be http(s) links")
        return value


class OwnerEmailContent(OwnerEmailModel):
    """The structured notice; ``commands`` are the owner-channel fallback
    commands shown in the muted footer (backticks render as code)."""

    heading: str = Field(min_length=1)
    intro: str | None = None
    details: tuple[OwnerEmailDetail, ...] = ()
    paragraphs: tuple[str, ...] = ()
    actions: tuple[OwnerEmailAction, ...] = ()
    action_note: str | None = None
    commands: tuple[str, ...] = ()
    footer: str | None = None
    business_name: str | None = None
    subject: str | None = None

    def with_actions(
        self, actions: tuple[OwnerEmailAction, ...], note: str | None
    ) -> "OwnerEmailContent":
        if not actions:
            return self
        return self.model_copy(update={"actions": actions, "action_note": note})

    def with_footer(self, footer: str | None) -> "OwnerEmailContent":
        return self.model_copy(update={"footer": footer})

    def email_subject(self) -> str:
        subject = (self.subject or self.heading)[:OWNER_EMAIL_SUBJECT_MAX_CHARS]
        return f"[{self.business_name}] {subject}" if self.business_name else subject


def owner_notice_content(
    text: str, *, business_name: str | None = None, subject: str | None = None
) -> OwnerEmailContent:
    """Any channel notice text as structured e-mail content.

    The first line's ``Heading — rest`` split gives the heading and intro;
    ``Label: value`` lines become details; lines with backticked commands go
    to the muted footer; the rest stay paragraphs.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = lines[0] if lines else "Notice"
    heading, dash, intro = first.partition(" — ")
    details: list[OwnerEmailDetail] = []
    paragraphs: list[str] = []
    commands: list[str] = []
    for line in lines[1:]:
        detail = _DETAIL_LINE.match(line)
        if "`" in line:
            commands.append(line)
        elif detail is not None:
            details.append(
                OwnerEmailDetail(
                    label=detail.group("label"), value=detail.group("value").rstrip(".")
                )
            )
        else:
            paragraphs.append(line)
    return OwnerEmailContent(
        heading=heading.strip() or first,
        intro=intro.strip() if dash and intro.strip() else None,
        details=tuple(details),
        paragraphs=tuple(paragraphs),
        commands=tuple(commands),
        business_name=business_name,
        subject=subject or first,
    )


def render_owner_email_text(content: OwnerEmailContent) -> str:
    blocks = [content.heading]
    if content.intro:
        blocks.append(content.intro)
    if content.details:
        blocks.append("\n".join(f"{item.label}: {item.value}" for item in content.details))
    blocks.extend(content.paragraphs)
    if content.actions:
        note = f"{content.action_note}\n" if content.action_note else ""
        blocks.append(note + "\n".join(f"{item.label}: {item.url}" for item in content.actions))
    if content.footer:
        blocks.append(content.footer)
    if content.commands:
        blocks.append("\n".join(content.commands))
    return "\n\n".join(blocks)


_FONT = (
    "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif,"
    "'Apple Color Emoji','Segoe UI Emoji'"
)
_BUTTON = (
    "display:inline-block;padding:10px 22px;margin:0 8px 8px 0;border-radius:6px;"
    "font-weight:600;font-size:15px;text-decoration:none;"
)
_PRIMARY = "background:#14532d;color:#ffffff;border:1px solid #14532d;"
_SECONDARY = "background:#ffffff;color:#7f1d1d;border:1px solid #b91c1c;"


def _inline(text: str) -> str:
    """Escaped text with ``code`` spans; nothing else is interpreted."""

    escaped = html.escape(text, quote=True)
    return _CODE.sub(
        lambda match: (
            '<code style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
            'font-size:13px;background:#f3f4f6;padding:1px 4px;border-radius:4px;">'
            f"{match.group(1)}</code>"
        ),
        escaped,
    )


def render_owner_email_html(content: OwnerEmailContent) -> str:
    """Self-contained HTML: inline styles, system font, no images or links
    other than the action buttons."""

    parts = [
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(content.heading)}</title></head>"
        f'<body style="margin:0;padding:24px 12px;background:#f6f7f9;font-family:{_FONT};'
        'color:#111827;">'
        '<div style="max-width:34rem;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;'
        'border-radius:10px;padding:24px 24px 16px;">'
    ]
    if content.business_name:
        parts.append(
            '<p style="margin:0 0 6px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;'
            f'color:#6b7280;">{html.escape(content.business_name)}</p>'
        )
    parts.append(
        '<h1 style="margin:0 0 12px;font-size:20px;line-height:1.3;font-weight:650;">'
        f"{_inline(content.heading)}</h1>"
    )
    if content.intro:
        parts.append(
            '<p style="margin:0 0 16px;font-size:15px;line-height:1.5;">'
            f"{_inline(content.intro)}</p>"
        )
    if content.details:
        parts.append(
            '<table role="presentation" style="width:100%;border-collapse:collapse;'
            'margin:0 0 16px;font-size:15px;line-height:1.45;">'
        )
        for item in content.details:
            parts.append(
                "<tr>"
                '<td style="padding:6px 12px 6px 0;color:#6b7280;vertical-align:top;'
                f'white-space:nowrap;width:1%;">{html.escape(item.label)}</td>'
                '<td style="padding:6px 0;vertical-align:top;">'
                f"{_inline(item.value)}</td></tr>"
            )
        parts.append("</table>")
    for paragraph in content.paragraphs:
        parts.append(
            f'<p style="margin:0 0 14px;font-size:15px;line-height:1.5;">{_inline(paragraph)}</p>'
        )
    if content.actions:
        parts.append('<div style="margin:8px 0 6px;">')
        for action in content.actions:
            style = _BUTTON + (_PRIMARY if action.primary else _SECONDARY)
            parts.append(
                f'<a href="{html.escape(action.url, quote=True)}" style="{style}">'
                f"{html.escape(action.label)}</a>"
            )
        parts.append("</div>")
        if content.action_note:
            parts.append(
                '<p style="margin:0 0 14px;font-size:13px;color:#6b7280;">'
                f"{_inline(content.action_note)}</p>"
            )
    if content.footer or content.commands:
        parts.append(
            '<div style="margin-top:18px;padding-top:12px;border-top:1px solid #e5e7eb;'
            'font-size:12.5px;line-height:1.55;color:#6b7280;">'
        )
        if content.footer:
            parts.append(f'<p style="margin:0 0 6px;">{_inline(content.footer)}</p>')
        for command in content.commands:
            parts.append(f'<p style="margin:0 0 6px;">{_inline(command)}</p>')
        parts.append("</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


class OwnerEmailRequest(OwnerEmailModel):
    """One owner e-mail send: multipart text + HTML."""

    business_id: BusinessId
    to: str = Field(min_length=3)
    subject: str = Field(min_length=1)
    text: str = Field(min_length=1)
    html: str | None = None
    idempotency_key: str = Field(min_length=1)


__all__ = [
    "OWNER_CHANNEL_FOOTER",
    "OWNER_EMAIL_SOURCE_NAMESPACE",
    "OWNER_EMAIL_SUBJECT_MAX_CHARS",
    "OwnerEmailAction",
    "OwnerEmailContent",
    "OwnerEmailDetail",
    "OwnerEmailRequest",
    "owner_notice_content",
    "render_owner_email_html",
    "render_owner_email_text",
]
