"""The CV advisor: few suggestions, no inventions, no deletions."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.cv.advisor import (
    Advice,
    Axis,
    Target,
    TargetKind,
    advise,
    apply_advice,
    load_advice_log,
    record_decision,
    validate_advice,
)
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.portals.form_learn import learn_form_html
from tests.fixtures.profile import (
    journalist_profile_dict,
    nurse_profile_dict,
    sample_profile_dict,
)


def _job(text: str, job_id: str = "J0001") -> JobPosting:
    return JobPosting.model_validate(
        {
            "id": job_id,
            "source": "indeed",
            "source_job_id": job_id,
            "title": "Un cargo",
            "company": "Una Empresa",
            "url": f"https://example.com/{job_id}",
            "description": text,
        }
    )


def test_it_returns_few_suggestions_per_run() -> None:
    """A list of thirty notes is a report nobody acts on."""
    candidate = Candidate.model_validate(sample_profile_dict())

    advice = advise(candidate, limit=3)

    assert len(advice) <= 3
    assert all(isinstance(item, Advice) for item in advice)


def test_glued_words_from_a_pdf_import_are_flagged_as_unreadable_by_machines() -> None:
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = (
        "Coordiné el turno de urgencias yReduje el tiempo de espera en 20 minutos."
    )
    candidate = Candidate.model_validate(raw)

    advice = advise(candidate, limit=10)
    glued = [a for a in advice if a.axis is Axis.MACHINE and "yReduje" in a.before]

    assert glued, "text with no space between words must be reported"
    assert glued[0].after and "yReduje" not in glued[0].after
    assert "y Reduje" in glued[0].after


def test_an_acronym_is_suggested_with_its_expansion_once() -> None:
    raw = journalist_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = (
        "Lideré la cobertura del CAE durante seis meses."
    )
    candidate = Candidate.model_validate(raw)

    advice = advise(candidate, limit=10)

    assert any(a.axis is Axis.MACHINE and "CAE" in a.what for a in advice)


def test_a_role_without_a_single_bullet_is_reported_as_layout() -> None:
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"] = []
    candidate = Candidate.model_validate(raw)

    advice = advise(candidate, limit=10)

    assert any(a.axis is Axis.LAYOUT for a in advice)


def test_an_overlong_bullet_is_suggested_to_be_split_not_deleted() -> None:
    raw = journalist_profile_dict()
    long_text = (
        "Coordiné la sección de actualidad con un equipo de ocho personas, definí la agenda "
        "semanal, negocié con las fuentes, revisé cada pieza antes de publicar, ajusté los "
        "titulares para buscadores y mantuve la relación con la audiencia en redes sociales "
        "durante todo el periodo del proyecto editorial."
    )
    raw["experience"][0]["achievements"][0]["text"] = long_text
    candidate = Candidate.model_validate(raw)

    advice = advise(candidate, limit=10)
    layout = [a for a in advice if a.axis is Axis.LAYOUT and a.before == long_text]

    assert layout
    assert not layout[0].after, "the advisor describes the problem; it never truncates a fact"


def test_market_language_only_rephrases_what_the_profile_already_backs() -> None:
    """The word may come from the market; the fact must come from the profile."""
    raw = nurse_profile_dict()
    candidate = Candidate.model_validate(raw)
    skill = candidate.skills.all_skills()[0]
    jobs = [
        _job(f"Buscamos experiencia en {skill}. Se valora {skill}.", f"J000{i}") for i in (1, 2)
    ]

    advice = advise(candidate, jobs=jobs, limit=10)

    for item in advice:
        if item.axis is Axis.LANGUAGE and item.after:
            assert validate_advice(item, candidate) is None


def test_a_form_question_becomes_a_question_never_an_answer(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/forms/greenhouse_apply.html").read_text(
        encoding="utf-8"
    )
    form = learn_form_html(html, url="https://boards.greenhouse.io/acme/jobs/4001")
    candidate = Candidate.model_validate(nurse_profile_dict())

    advice = advise(candidate, forms=[form], limit=10)
    from_form = [a for a in advice if "form" in a.why.casefold()]

    assert from_form, "what employers ask on their forms must reach the advisor"
    assert all(not a.after for a in from_form), "a question is asked, not answered for you"


def test_a_rewrite_that_invents_a_tool_is_rejected() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    text = candidate.experience[0].achievements[0].text

    invented = Advice(
        axis=Axis.LANGUAGE,
        target=Target(
            kind=TargetKind.ACHIEVEMENT,
            experience_id=candidate.experience[0].id,
            achievement_id=candidate.experience[0].achievements[0].id,
        ),
        what="reword it",
        before=text,
        after=f"{text} Implementado con Kubernetes y Terraform.",
    )

    reason = validate_advice(invented, candidate)

    assert reason is not None
    assert "kubernetes" in reason.casefold() or "terraform" in reason.casefold()


def test_a_rewrite_that_drops_a_number_is_rejected() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    exp = candidate.experience[0]
    before = "Reduje el tiempo de espera en 20 minutos con el equipo de 8 personas."
    after = "Reduje el tiempo de espera trabajando con el equipo."

    reason = validate_advice(
        Advice(
            axis=Axis.LANGUAGE,
            target=Target(
                kind=TargetKind.ACHIEVEMENT,
                experience_id=exp.id,
                achievement_id=exp.achievements[0].id,
            ),
            what="tighten it",
            before=before,
            after=after,
        ),
        candidate,
    )

    assert reason is not None
    assert "20" in reason


def test_a_rewrite_that_only_changes_wording_is_accepted() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    exp = candidate.experience[0]
    before = "Fui responsable de coordinar el turno de urgencias con 8 personas."
    after = "Coordiné el turno de urgencias con 8 personas."

    reason = validate_advice(
        Advice(
            axis=Axis.LANGUAGE,
            target=Target(
                kind=TargetKind.ACHIEVEMENT,
                experience_id=exp.id,
                achievement_id=exp.achievements[0].id,
            ),
            what="lead with the verb",
            before=before,
            after=after,
        ),
        candidate,
    )

    assert reason is None


def test_applying_a_rewrite_touches_only_that_text() -> None:
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = "Fui responsable de coordinar el turno."
    exp_id = raw["experience"][0]["id"]
    ach_id = raw["experience"][0]["achievements"][0]["id"]
    advice = Advice(
        axis=Axis.LANGUAGE,
        target=Target(kind=TargetKind.ACHIEVEMENT, experience_id=exp_id, achievement_id=ach_id),
        what="lead with the verb",
        before="Fui responsable de coordinar el turno.",
        after="Coordiné el turno.",
    )

    assert apply_advice(raw, advice)
    assert raw["experience"][0]["achievements"][0]["text"] == "Coordiné el turno."
    assert raw["personal"]["name"] == nurse_profile_dict()["personal"]["name"]


def test_applying_a_stale_rewrite_changes_nothing() -> None:
    """If the text moved on, the old proposal must not overwrite it."""
    raw = nurse_profile_dict()
    exp_id = raw["experience"][0]["id"]
    ach_id = raw["experience"][0]["achievements"][0]["id"]
    advice = Advice(
        axis=Axis.LANGUAGE,
        target=Target(kind=TargetKind.ACHIEVEMENT, experience_id=exp_id, achievement_id=ach_id),
        what="reword",
        before="Un texto que ya no está en el perfil.",
        after="Otro texto.",
    )

    assert not apply_advice(raw, advice)


def test_a_rejected_suggestion_is_not_proposed_again(tmp_path: Path) -> None:
    """The log exists so a run proposes something new instead of repeating itself."""
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"] = []
    candidate = Candidate.model_validate(raw)
    log_path = tmp_path / "advice_log.yaml"

    first = advise(candidate, limit=1, log_path=log_path)
    assert first
    record_decision(log_path, first[0], "rejected")

    again = advise(candidate, limit=1, log_path=log_path)

    assert all(item.id != first[0].id for item in again)
    log = load_advice_log(log_path)
    assert log[first[0].id] == "rejected"


def test_an_applied_suggestion_is_not_proposed_again(tmp_path: Path) -> None:
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = "Fui responsable de coordinar el turno."
    candidate = Candidate.model_validate(raw)
    log_path = tmp_path / "advice_log.yaml"

    first = advise(candidate, limit=1, log_path=log_path)
    assert first, "a weak opener is something to improve"
    record_decision(log_path, first[0], "applied")

    assert all(item.id != first[0].id for item in advise(candidate, limit=5, log_path=log_path))


def test_the_same_problem_gets_the_same_id_across_runs() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())

    first = advise(candidate, limit=5)
    second = advise(candidate, limit=5)

    assert [a.id for a in first] == [a.id for a in second]


@pytest.mark.parametrize(
    "profile_dict",
    [nurse_profile_dict, journalist_profile_dict, sample_profile_dict],
)
def test_it_advises_any_profession_the_same_way(profile_dict: object) -> None:
    """No axis may depend on the vocabulary of one field."""
    candidate = Candidate.model_validate(profile_dict())  # type: ignore[operator]

    advice = advise(candidate, limit=5)

    for item in advice:
        assert item.what
        assert item.why
        if item.after:
            assert validate_advice(item, candidate) is None


def test_nothing_is_ever_proposed_for_deletion() -> None:
    """Whatever the run says, no suggestion may empty an existing text."""
    for maker in (nurse_profile_dict, journalist_profile_dict, sample_profile_dict):
        candidate = Candidate.model_validate(maker())
        for item in advise(candidate, limit=20):
            if item.before:
                assert item.after != ""or not item.is_rewrite
                assert item.after is not None


def test_a_product_name_in_camel_case_is_not_a_missing_space() -> None:
    """Found on a real profile: it proposed 'Google Big Query' and 'Analytics Base Table'."""
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = (
        "Construí el registro clínico sobre Google BigQuery con un AnalyticsBaseTable propio."
    )
    candidate = Candidate.model_validate(raw)

    advice = advise(candidate, limit=20)
    machine = [a for a in advice if a.axis is Axis.MACHINE and "separate" in a.what]

    assert machine == [], f"camelCase names must be left alone: {[a.what for a in machine]}"


def test_a_word_glued_to_the_next_one_is_still_caught() -> None:
    """The real accident: the space glyph is gone between two ordinary words."""
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = (
        "Coordiné el turno de urgencias yReduje la espera; con experienciaClínica acreditada."
    )
    candidate = Candidate.model_validate(raw)

    flagged = [
        a.what
        for a in advise(candidate, limit=20)
        if a.axis is Axis.MACHINE and "separate" in a.what
    ]

    assert any("yReduje" in what for what in flagged)
    assert any("experienciaClínica" in what for what in flagged)


def test_a_shorter_rewrite_that_keeps_facts_is_accepted() -> None:
    """Density: same numbers and names, fewer words (#54)."""
    candidate = Candidate.model_validate(nurse_profile_dict())
    exp = candidate.experience[0]
    before = (
        "Coordiné el turno de urgencias con el fin de reducir la espera en 20 minutos "
        "en el Hospital Central."
    )
    after = (
        "Coordiné el turno de urgencias para reducir la espera en 20 minutos "
        "en el Hospital Central."
    )

    reason = validate_advice(
        Advice(
            axis=Axis.LANGUAGE,
            target=Target(
                kind=TargetKind.ACHIEVEMENT,
                experience_id=exp.id,
                achievement_id=exp.achievements[0].id,
            ),
            what="same facts, fewer words",
            before=before,
            after=after,
        ),
        candidate,
    )

    assert reason is None


def test_a_longer_rewrite_without_backed_market_term_is_rejected() -> None:
    """Lengthening without a market term the profile/JD already backs is fluff."""
    candidate = Candidate.model_validate(nurse_profile_dict())
    exp = candidate.experience[0]
    before = "Coordiné el turno de urgencias con 8 personas."
    after = (
        "Coordiné de manera excepcional el turno de urgencias con 8 personas "
        "en un entorno altamente desafiante."
    )

    reason = validate_advice(
        Advice(
            axis=Axis.LANGUAGE,
            target=Target(
                kind=TargetKind.ACHIEVEMENT,
                experience_id=exp.id,
                achievement_id=exp.achievements[0].id,
            ),
            what="decorate it",
            before=before,
            after=after,
        ),
        candidate,
    )

    assert reason is not None


def test_intensifier_only_expansion_is_rejected() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    exp = candidate.experience[0]
    before = "Coordiné el turno de urgencias."
    after = "Coordiné realmente el turno de urgencias de forma altamente efectiva."

    reason = validate_advice(
        Advice(
            axis=Axis.LANGUAGE,
            target=Target(
                kind=TargetKind.ACHIEVEMENT,
                experience_id=exp.id,
                achievement_id=exp.achievements[0].id,
            ),
            what="intensify",
            before=before,
            after=after,
        ),
        candidate,
    )

    assert reason is not None
    folded = reason.casefold()
    assert "fluff" in folded or "intensif" in folded or "longer" in folded


def test_compress_redundant_phrases_yields_shorter_advice() -> None:
    from jobbot.cv.advisor import compress_phrase

    before = "Lideré el proyecto con el fin de mejorar la cobertura en orden de prioridad."
    after = compress_phrase(before)

    assert after is not None
    assert len(after) < len(before)
    assert "con el fin de" not in after.casefold()
    assert "para" in after.casefold()


def test_advise_prefers_shorter_rewrites_first() -> None:
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = (
        "Fui responsable de coordinar el turno con el fin de reducir la espera en 20 minutos."
    )
    raw["experience"][0]["achievements"].append(
        {
            "id": "nurse-extra-long",
            "text": (
                "También fui responsable de documentar los protocolos "
                "con el fin de estandarizar el turno."
            ),
            "tags": [],
            "metrics": {},
        }
    )
    candidate = Candidate.model_validate(raw)

    advice = advise(candidate, limit=5)
    rewrites = [a for a in advice if a.is_rewrite]
    assert rewrites
    deltas = [len(a.after) - len(a.before) for a in rewrites]
    assert deltas == sorted(deltas), "shorter (more negative delta) must come first"


def test_advise_respects_budget_when_scoped_to_one_job() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    skill = candidate.skills.all_skills()[0]
    job = _job(f"Se requiere {skill}. Experiencia en {skill} es clave.", "J0099")

    advice = advise(candidate, jobs=[job], limit=3)

    assert len(advice) <= 3


def test_char_delta_reports_how_much_tighter() -> None:
    from jobbot.cv.advisor import char_delta

    assert char_delta("abcdefghij", "abcd") == -6
    assert char_delta("hi", "hello") == 3
