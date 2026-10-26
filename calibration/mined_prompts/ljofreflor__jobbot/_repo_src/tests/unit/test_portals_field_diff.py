"""Field diff: discover what forms ask that profile.yaml doesn't have."""

from __future__ import annotations

from jobbot.portals.field_diff import (
    NewFieldCandidate,
    diff_form_fields,
    field_semantic_hash,
    has_semantic_match,
    normalize_field_label,
)
from jobbot.portals.form_learn import FieldKind, FormField, FormKnowledge


def test_normalize_field_label_handles_synonyms() -> None:
    assert normalize_field_label("email") == "email"
    assert normalize_field_label("E-mail") == "email"
    assert normalize_field_label("correo electrónico") == "email"
    assert normalize_field_label("Correo") == "email"
    
    assert normalize_field_label("teléfono") == "phone"
    assert normalize_field_label("celular") == "phone"
    assert normalize_field_label("móvil") == "phone"
    
    assert normalize_field_label("curriculum") == "cv"
    assert normalize_field_label("resume") == "cv"
    assert normalize_field_label("hoja de vida") == "cv"


def test_field_semantic_hash_same_for_synonyms() -> None:
    hash1 = field_semantic_hash("email", FieldKind.EMAIL)
    hash2 = field_semantic_hash("E-mail", FieldKind.EMAIL)
    hash3 = field_semantic_hash("correo electrónico", FieldKind.EMAIL)
    
    assert hash1 == hash2 == hash3


def test_field_semantic_hash_different_for_different_fields() -> None:
    email_hash = field_semantic_hash("email", FieldKind.EMAIL)
    phone_hash = field_semantic_hash("phone", FieldKind.PHONE)
    
    assert email_hash != phone_hash


def test_has_semantic_match_recognizes_known_fields() -> None:
    known = {
        field_semantic_hash("email", FieldKind.EMAIL),
        field_semantic_hash("phone", FieldKind.PHONE),
    }
    
    assert has_semantic_match("email", FieldKind.EMAIL, known)
    assert has_semantic_match("correo electrónico", FieldKind.EMAIL, known)
    assert has_semantic_match("teléfono", FieldKind.PHONE, known)
    
    assert not has_semantic_match("work authorization", FieldKind.SELECT, known)


def test_diff_form_fields_finds_new_fields() -> None:
    forms = [
        FormKnowledge(
            url="https://greenhouse.io/apply",
            company="Acme",
            readable=True,
            fields=[
                FormField(
                    name="email",
                    label="Email",
                    kind=FieldKind.EMAIL,
                    required=True,
                    is_screening=False,
                ),
                FormField(
                    name="work_auth",
                    label="Work authorization Chile",
                    kind=FieldKind.SELECT,
                    required=True,
                    options=["Sí", "No"],
                    is_screening=True,
                ),
            ],
        ),
        FormKnowledge(
            url="https://workday.com/apply",
            company="RetailCo",
            readable=True,
            fields=[
                FormField(
                    name="email",
                    label="Correo electrónico",
                    kind=FieldKind.EMAIL,
                    required=True,
                    is_screening=False,
                ),
                FormField(
                    name="work_auth_cl",
                    label="Work authorization Chile",
                    kind=FieldKind.SELECT,
                    required=True,
                    options=["Yes", "No", "In process"],
                    is_screening=True,
                ),
            ],
        ),
    ]
    
    candidates = diff_form_fields(forms, min_frequency=1)
    
    # Email should NOT appear (it's known)
    assert not any("email" in c.labels_seen[0].lower() for c in candidates)
    
    # Work authorization SHOULD appear (it's new) and seen in both portals
    work_auth = next((c for c in candidates if "work" in c.labels_seen[0].lower()), None)
    assert work_auth is not None
    assert work_auth.frequency == 2  # Seen in 2 portals
    assert work_auth.kind == FieldKind.SELECT
    assert work_auth.required_ratio == 1.0  # Both marked as required


def test_diff_form_fields_respects_min_frequency() -> None:
    forms = [
        FormKnowledge(
            url="https://greenhouse.io/apply",
            company="Acme",
            readable=True,
            fields=[
                FormField(
                    name="rare_field",
                    label="Very specific question only Acme asks",
                    kind=FieldKind.TEXT,
                    required=False,
                    is_screening=True,
                ),
            ],
        ),
        FormKnowledge(
            url="https://workday.com/apply",
            company="RetailCo",
            readable=True,
            fields=[
                FormField(
                    name="common_field",
                    label="Common question multiple portals ask",
                    kind=FieldKind.SELECT,
                    required=True,
                    is_screening=True,
                ),
            ],
        ),
        FormKnowledge(
            url="https://lever.co/apply",
            company="StartupCo",
            readable=True,
            fields=[
                FormField(
                    name="common_field",
                    label="Common question multiple portals ask",
                    kind=FieldKind.SELECT,
                    required=True,
                    is_screening=True,
                ),
            ],
        ),
    ]
    
    # With min_frequency=1, both should appear
    candidates_1 = diff_form_fields(forms, min_frequency=1)
    assert len(candidates_1) == 2
    
    # With min_frequency=2, only common field should appear
    candidates_2 = diff_form_fields(forms, min_frequency=2)
    assert len(candidates_2) == 1
    assert "common question" in candidates_2[0].labels_seen[0].lower()


def test_diff_form_fields_skips_unreadable_forms() -> None:
    forms = [
        FormKnowledge(
            url="https://example.com/apply",
            company="Acme",
            readable=False,
            evidence="client-rendered",
            fields=[],
        ),
        FormKnowledge(
            url="https://example2.com/apply",
            company="RetailCo",
            readable=True,
            fields=[
                FormField(
                    name="real_field",
                    label="Real field from readable form",
                    kind=FieldKind.TEXT,
                    required=True,
                    is_screening=True,
                ),
            ],
        ),
    ]
    
    candidates = diff_form_fields(forms, min_frequency=1)
    
    # Only the readable form's field should appear
    assert len(candidates) == 1
    assert "real field" in candidates[0].labels_seen[0].lower()


def test_new_field_candidate_properties() -> None:
    candidate = NewFieldCandidate(
        field_hash="abc123",
        labels_seen=["Work authorization", "¿Visa de trabajo?"],
        kind=FieldKind.SELECT,
        portals=["greenhouse.io", "workday.com", "lever.co"],
        required_count=2,
        total_count=3,
        options=["Yes", "No"],
        example_accepts=[],
    )
    
    assert candidate.frequency == 3
    assert candidate.required_ratio == 2 / 3
    assert candidate.is_common is True


def test_diff_form_fields_sorts_by_frequency() -> None:
    forms = [
        FormKnowledge(
            url="https://a.com/apply",
            company="A",
            readable=True,
            fields=[
                FormField(
                    name="common",
                    label="Very common field",
                    kind=FieldKind.TEXT,
                    is_screening=True,
                ),
            ],
        ),
        FormKnowledge(
            url="https://b.com/apply",
            company="B",
            readable=True,
            fields=[
                FormField(
                    name="common",
                    label="Very common field",
                    kind=FieldKind.TEXT,
                    is_screening=True,
                ),
                FormField(
                    name="rare",
                    label="Rare field",
                    kind=FieldKind.TEXT,
                    is_screening=True,
                ),
            ],
        ),
        FormKnowledge(
            url="https://c.com/apply",
            company="C",
            readable=True,
            fields=[
                FormField(
                    name="common",
                    label="Very common field",
                    kind=FieldKind.TEXT,
                    is_screening=True,
                ),
            ],
        ),
    ]
    
    candidates = diff_form_fields(forms, min_frequency=1)
    
    # Most common field should be first
    assert len(candidates) >= 2
    assert candidates[0].frequency >= candidates[1].frequency
