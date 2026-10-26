"""What an application form asks, learned without submitting or storing answers."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.portals.detect import AtsKind
from jobbot.portals.form_learn import (
    FieldKind,
    FormKnowledge,
    default_form_knowledge_path,
    learn_form_html,
    load_form_knowledge,
    save_form_knowledge,
    upsert_form,
)

FIXTURE = Path("tests/fixtures/forms/greenhouse_apply.html")
APPLY_URL = "https://boards.greenhouse.io/acme/jobs/4001"


def _learned(project_root: Path) -> FormKnowledge:
    html = (project_root / FIXTURE).read_text(encoding="utf-8")
    return learn_form_html(html, url=APPLY_URL, company="Acme")


def test_it_reads_which_fields_the_form_asks_for(project_root: Path) -> None:
    form = _learned(project_root)

    labels = [field.label for field in form.fields]
    assert "First Name" in labels
    assert "Email" in labels
    assert "Resume/CV" in labels
    assert form.readable
    assert form.ats is AtsKind.GREENHOUSE
    assert form.company == "Acme"


def test_it_records_kind_and_whether_the_field_is_required(project_root: Path) -> None:
    by_label = {field.label: field for field in _learned(project_root).fields}

    assert by_label["Email"].kind is FieldKind.EMAIL
    assert by_label["Email"].required
    assert by_label["Phone"].kind is FieldKind.PHONE
    assert not by_label["Phone"].required
    assert by_label["Resume/CV"].kind is FieldKind.FILE
    assert by_label["LinkedIn Profile"].kind is FieldKind.URL
    assert by_label["Expected monthly salary (CLP)"].kind is FieldKind.NUMBER
    assert by_label["I agree to the privacy policy"].kind is FieldKind.CHECKBOX


def test_it_keeps_the_choices_a_select_offers(project_root: Path) -> None:
    by_label = {field.label: field for field in _learned(project_root).fields}

    english = by_label["Nivel de inglés"]
    assert english.kind is FieldKind.SELECT
    assert english.options == ["Básico", "Intermedio", "Avanzado"]
    assert "Selecciona" not in english.options, "the empty prompt is not a choice"

    radio = by_label["How did you hear about this job?"]
    assert radio.kind is FieldKind.RADIO
    assert radio.options == ["Referral", "LinkedIn", "Other"]


def test_screening_questions_are_marked_apart_from_identity_fields(
    project_root: Path,
) -> None:
    """The questions are the part the CV has to answer; a name is just a name."""
    by_label = {field.label: field for field in _learned(project_root).fields}

    assert by_label["Are you legally authorized to work in Chile?"].is_screening
    assert by_label["When could you start?"].is_screening
    assert by_label["Expected monthly salary (CLP)"].is_screening
    assert not by_label["First Name"].is_screening
    assert not by_label["Email"].is_screening

    questions = _learned(project_root).screening_questions()
    assert "When could you start?" in questions


def test_no_answer_is_ever_stored(project_root: Path) -> None:
    """The form is read for its questions; the values on it belong to whoever typed them."""
    form = _learned(project_root)
    dumped = form.model_dump_json()

    assert "Ada" not in dumped
    assert "ada@example.com" not in dumped
    assert "do-not-store-me" not in dumped
    assert not any(hasattr(field, "value") for field in form.fields)


def test_placeholder_examples_are_kept_free_of_contact_data(project_root: Path) -> None:
    """A placeholder is a hint about the format, and sometimes a real phone."""
    dumped = _learned(project_root).model_dump_json()

    assert "+56 9 1234 5678" not in dumped
    assert "tu.correo@example.com" not in dumped


def test_a_hidden_technical_field_is_not_a_question(project_root: Path) -> None:
    names = [field.name for field in _learned(project_root).fields]

    assert not any("authenticity_token" in name for name in names)


def test_a_form_it_cannot_read_is_reported_as_unknown() -> None:
    """A client-rendered page gives us nothing; that is not an empty form."""
    form = learn_form_html(
        "<html><body><div id='root'></div></body></html>",
        url=APPLY_URL,
    )

    assert form.fields == []
    assert not form.readable
    assert "unknown" in form.evidence.casefold()


def test_a_page_with_only_a_search_box_is_not_an_application_form() -> None:
    html = (
        "<html><body><form action='/search'>"
        "<label for='q'>Buscar</label><input type='search' id='q' name='q'>"
        "</form></body></html>"
    )

    assert not learn_form_html(html, url="https://empresa.cl/empleos").readable


def test_knowledge_round_trips_and_dedups_by_url(tmp_path: Path, project_root: Path) -> None:
    path = tmp_path / "form_knowledge.yaml"
    first = _learned(project_root)

    forms = upsert_form([], first)
    forms = upsert_form(forms, _learned(project_root))
    save_form_knowledge(forms, path)

    assert len(forms) == 1, "the same form observed twice is one entry"
    reloaded = load_form_knowledge(path)
    assert len(reloaded) == 1
    assert reloaded[0].url == first.url
    assert [f.label for f in reloaded[0].fields] == [f.label for f in first.fields]


def test_tracking_parameters_do_not_create_a_second_entry(project_root: Path) -> None:
    html = (project_root / FIXTURE).read_text(encoding="utf-8")
    plain = learn_form_html(html, url=APPLY_URL)
    tracked = learn_form_html(html, url=f"{APPLY_URL}?utm_source=linkedin&gh_src=abc")

    assert len(upsert_form([plain], tracked)) == 1


def test_what_it_learned_stays_a_candidate_until_you_look(project_root: Path) -> None:
    assert _learned(project_root).status.value == "candidate"


def test_missing_file_reads_as_no_knowledge(tmp_path: Path) -> None:
    assert load_form_knowledge(tmp_path / "nope.yaml") == []


def test_the_knowledge_file_lives_where_git_ignores_it() -> None:
    path = default_form_knowledge_path(Path("/tmp/root"))

    assert path.name == "form_knowledge.yaml"
    assert path.parent.name == "data"


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("¿Cuál es tu expectativa de renta?", FieldKind.TEXT),
        ("Why do you want to work here?", FieldKind.TEXT),
    ],
)
def test_a_free_text_question_is_still_a_question(label: str, kind: FieldKind) -> None:
    html = (
        "<html><body><form action='/apply'>"
        "<label for='cv'>CV</label><input type='file' id='cv' name='cv' required>"
        f"<label for='q1'>{label}</label><input type='text' id='q1' name='q1'>"
        "</form></body></html>"
    )

    fields = {field.label: field for field in learn_form_html(html, url=APPLY_URL).fields}

    assert fields[label].kind is kind
    assert fields[label].is_screening
