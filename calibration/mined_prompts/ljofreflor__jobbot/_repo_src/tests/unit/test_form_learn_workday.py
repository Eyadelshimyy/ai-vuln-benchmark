"""Workday Create Account uses type=text for email (#90)."""

from __future__ import annotations

from pathlib import Path

from jobbot.portals.form_learn import FieldKind, learn_form_html
from jobbot.portals.sso import detect_sso_providers

FIXTURE = Path("tests/fixtures/forms/workday_create_account.html")
URL = (
    "https://example.wd5.myworkdayjobs.com/en-US/External_Career/"
    "job/Remote/Role_1/apply/applyManually"
)


def test_workday_create_account_email_type_text_is_learned(project_root: Path) -> None:
    html = (project_root / FIXTURE).read_text(encoding="utf-8")

    form = learn_form_html(html, url=URL, company="Acme")

    assert form.readable, form.evidence
    by_label = {field.label: field for field in form.fields}
    assert "Email Address" in by_label
    assert by_label["Email Address"].kind is FieldKind.EMAIL
    assert by_label["Email Address"].required
    assert not any("password" in f.label.casefold() for f in form.fields)
    assert detect_sso_providers(html) == ()
