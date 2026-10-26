"""Portal label → profile fact homologation (#94)."""

from __future__ import annotations

from jobbot.models.candidate import Candidate
from jobbot.portals.field_homologation import (
    ProfileFact,
    aliases_for,
    answer_for_label,
    resolve_fact,
)
from jobbot.portals.form_learn import FieldKind, FormField, FormKnowledge
from tests.fixtures.profile import sample_profile_dict


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def test_email_address_resolves_to_email_fact() -> None:
    assert resolve_fact("Email Address*") is ProfileFact.EMAIL
    assert resolve_fact("correo electrónico") is ProfileFact.EMAIL
    value, source = answer_for_label(_candidate(), "Email Address")
    assert value == str(_candidate().personal.email)
    assert "email" in source


def test_middle_name_and_fathers_family_are_not_homologated() -> None:
    """Those Workday labels have no profile fact — leave empty, do not guess."""
    assert resolve_fact("Middle Name") is None
    assert resolve_fact("Father's Family Name") is None
    assert resolve_fact("Mother's Family Name") is None
    value, source = answer_for_label(_candidate(), "Middle Name")
    assert value == ""
    assert "you decide" in source.casefold()


def test_bare_name_is_full_name_not_middle_name() -> None:
    assert resolve_fact("Name") is ProfileFact.FULL_NAME
    assert resolve_fact("Middle Name") is not ProfileFact.FULL_NAME


def test_given_name_aliases_include_workday_wording() -> None:
    aliases = aliases_for(ProfileFact.GIVEN_NAME)
    assert any("given name" in a for a in aliases)
    assert resolve_fact("Given Name(s)") is ProfileFact.GIVEN_NAME


def test_signup_sheet_uses_homologation_not_substring() -> None:
    from jobbot.companies.signup import signup_sheet

    form = FormKnowledge(
        url="https://example.wd5.myworkdayjobs.com/x",
        readable=True,
        fields=[
            FormField(name="middle", label="Middle Name", kind=FieldKind.TEXT),
            FormField(name="father", label="Father's Family Name", kind=FieldKind.TEXT),
            FormField(name="email", label="Email Address", kind=FieldKind.EMAIL),
            FormField(name="given", label="Given Name(s)", kind=FieldKind.TEXT),
            FormField(name="city", label="City", kind=FieldKind.TEXT),
        ],
    )
    sheet = {item.label: item for item in signup_sheet(_candidate(), form=form)}

    assert sheet["Middle Name"].value == ""
    assert sheet["Father's Family Name"].value == ""
    assert sheet["Email Address"].value == str(_candidate().personal.email)
    assert sheet["Given Name(s)"].value
    assert sheet["City"].value == (_candidate().personal.city or "")
