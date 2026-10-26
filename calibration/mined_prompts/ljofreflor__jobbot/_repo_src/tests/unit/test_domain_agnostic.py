"""JobBot must serve any candidate, not only a data scientist.

Every module that reads a job description or a profile used to carry a closed list
of data-science vocabulary, so a nurse or a journalist got empty skills, a zero
match and a CV with no selected evidence. One test per module, all with profiles
whose words never appear in a data stack.
"""

from __future__ import annotations

from pathlib import Path

from jobbot.adapters.getonboard.draft import (
    PermanentProfileFields,
    build_permanent_profile_fields,
)
from jobbot.adapters.getonboard.package import (
    build_getonboard_sync_package,
    render_getonboard_sync_markdown,
)
from jobbot.adapters.linkedin.sweep import looks_like_job_post
from jobbot.cv.renderer import render_cv_tex
from jobbot.cv.selection import select_for_job
from jobbot.jobs.parsing import parse_job_text
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.models.candidate import Candidate
from jobbot.models.match import MatchStrength
from jobbot.nlp.refine import CumulativeProfileRefiner
from jobbot.profile.importer_pdf import parse_cv_text
from jobbot.profile.market import suggest_from_market
from tests.fixtures.profile import journalist_profile_dict, nurse_profile_dict

NURSE_JOB = (
    "Title: Enfermera Clínica Unidad de Paciente Crítico\n"
    "Company: Clínica Cordillera\n"
    "Location: Santiago, Chile\n"
    "Requisitos:\n"
    "- Título de Enfermera Universitaria con registro vigente.\n"
    "- Experiencia en ventilación mecánica y fármacos vasoactivos.\n"
    "- Manejo de reanimación cardiopulmonar avanzada.\n"
    "- Registro clínico en ficha electrónica.\n"
)

JOURNALIST_JOB = (
    "Title: Editor de Contenidos Digitales\n"
    "Company: Medio Regional\n"
    "Location: Concepción, Chile\n"
    "Requisitos:\n"
    "- Título de Periodista.\n"
    "- Edición digital y redacción periodística de cierre diario.\n"
    "- Conocimiento de SEO y manejo de WordPress.\n"
)


def test_job_skills_are_extracted_outside_the_data_vocabulary() -> None:
    """A clinical JD used to yield skills=[] because the hint list was DS-only."""
    job = parse_job_text(NURSE_JOB, job_id="J0600")

    folded = " ".join(job.skills).casefold()
    assert "ventilación mecánica" in folded
    assert "reanimación cardiopulmonar" in folded
    assert job.requirements


def test_newsroom_tools_are_extracted_as_skills() -> None:
    job = parse_job_text(JOURNALIST_JOB, job_id="J0601")

    folded = " ".join(job.skills).casefold()
    assert "seo" in folded
    assert "wordpress" in folded


def test_nurse_matches_a_nursing_job() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    job = parse_job_text(NURSE_JOB, job_id="J0602")

    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    strong = " ".join(i.label.casefold() for i in match.by_strength(MatchStrength.STRONG))

    assert match.score >= 60
    assert "ventilación mecánica" in strong
    role = next(i for i in match.items if i.label.startswith("role:"))
    assert role.strength == MatchStrength.STRONG


def test_journalist_matches_a_newsroom_job() -> None:
    candidate = Candidate.model_validate(journalist_profile_dict())
    job = parse_job_text(JOURNALIST_JOB, job_id="J0603")

    match = RuleBasedJobAnalyzer().analyze(candidate, job)

    assert match.score >= 60


def test_nurse_does_not_match_a_data_science_job() -> None:
    """Domain-agnostic cuts both ways: no free credit for an unrelated role."""
    candidate = Candidate.model_validate(nurse_profile_dict())
    job = parse_job_text(
        "Title: Senior Data Scientist\nCompany: NeuralWorks\n"
        "Requirements:\n- Python\n- SQL\n- Machine Learning\n",
        job_id="J0604",
    )

    match = RuleBasedJobAnalyzer().analyze(candidate, job)

    assert match.score < 40


def test_cv_selection_cites_clinical_evidence_instead_of_falling_back() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    job = parse_job_text(NURSE_JOB, job_id="J0605")

    selection = select_for_job(candidate, job)
    reasons = " ".join(r for a in selection.selected_achievements for r in a.reason)

    assert "fallback" not in reasons
    assert "ventilación mecánica" in reasons.casefold()
    assert "Ventilación Mecánica" in selection.skill_names


def test_market_terms_come_from_the_stored_jobs_not_from_a_data_list() -> None:
    candidate = Candidate.model_validate(nurse_profile_dict())
    jobs = [
        parse_job_text(NURSE_JOB, job_id="J0606"),
        parse_job_text(NURSE_JOB.replace("Cordillera", "Los Andes"), job_id="J0607"),
    ]

    suggestion = suggest_from_market(candidate, jobs)
    terms = " ".join(term for term, _ in suggestion.market_terms).casefold()

    assert "ventilación mecánica" in terms
    assert "machine learning" not in terms
    assert any("ventilación" in term.casefold() for term in suggestion.present)


def test_sweep_keeps_a_hiring_post_from_any_field() -> None:
    """The post filter used to require data-science words to consider a post at all."""
    clinical = (
        "Buscamos Enfermera Clínica para Unidad de Paciente Crítico en Santiago. "
        "Enviar CV a seleccion@example.com."
    )
    newsroom = "Vacante: Editor de Contenidos Digitales. Postula en nuestro portal."
    chatter = "Gran fin de semana de trekking en el cajón del Maipo."

    assert looks_like_job_post(clinical)
    assert looks_like_job_post(newsroom)
    assert not looks_like_job_post(chatter)


def test_linkedin_message_apply_is_a_channel_not_a_trade() -> None:
    """A clinic or a newsroom can ask for a DM; the detector must not require a stack."""
    from jobbot.portals.message_apply import asks_for_linkedin_message

    clinical = (
        "Seguimos buscando un(a) Enfermera Clínica | Chile | Híbrido. "
        "Si te interesa, mandame un mensaje indicando que te interesa este puesto."
    )
    chatter = "Te mandé un mensaje ayer sobre el trekking en el cajón del Maipo."

    assert looks_like_job_post(clinical)
    assert asks_for_linkedin_message(clinical)
    assert not asks_for_linkedin_message(chatter)
    assert not looks_like_job_post(chatter)


def test_refine_keeps_a_paragraph_backed_by_a_non_data_profile() -> None:
    """The old rule kept a paragraph only if it mentioned data-science words."""
    candidate = Candidate.model_validate(nurse_profile_dict())
    previous = PermanentProfileFields(
        experiencia_y_perfil=(
            "Enfermera clínica en Hospital del Puerto, unidad de paciente crítico "
            "adulto, con foco en ventilación mecánica.\n\n"
            "Trabajé en Laboratorio Antiguo operando un equipo que ya no uso."
        ),
        formacion_academica="Enfermera Universitaria, Universidad de Valparaíso.",
        headline="Enfermera Clínica",
        skills=["Ventilación Mecánica"],
        signature=None,
    )

    result = CumulativeProfileRefiner().refine_with_stats(candidate, previous)

    assert "paciente crítico" in result.fields.experiencia_y_perfil.casefold()
    assert "Laboratorio Antiguo" not in result.fields.experiencia_y_perfil
    assert result.kept_paragraphs >= 1
    assert result.dropped_paragraphs >= 1


def test_refine_keeps_a_generic_paragraph_with_no_unbacked_employer() -> None:
    """A paragraph was kept only if it named data-science words; any other field lost it."""
    candidate = Candidate.model_validate(nurse_profile_dict())
    previous = PermanentProfileFields(
        experiencia_y_perfil=(
            "Lidero la capacitación de personal en turnos de urgencia y "
            "coordino la entrega de turno."
        ),
        formacion_academica="Enfermera Universitaria.",
        headline="Enfermera Clínica",
        skills=["Ventilación Mecánica"],
        signature=None,
    )

    result = CumulativeProfileRefiner().refine_with_stats(candidate, previous)

    assert "turnos de urgencia" in result.fields.experiencia_y_perfil
    assert result.dropped_paragraphs == 0


def test_cv_section_labels_do_not_assert_a_field(project_root: Path) -> None:
    """'Habilidades Técnicas' and 'Publicaciones Científicas' named a field for everyone."""
    candidate = Candidate.model_validate(nurse_profile_dict())

    tex = render_cv_tex(candidate, project_root / "templates")

    assert r"\section{Habilidades}" in tex
    assert "Habilidades Técnicas" not in tex
    assert "Publicaciones Científicas" not in tex


def test_portal_texts_carry_no_other_persons_name() -> None:
    """The GoB signature and the CV-naming hint used to name the tool's author."""
    candidate = Candidate.model_validate(nurse_profile_dict())

    fields = build_permanent_profile_fields(candidate)
    package = build_getonboard_sync_package(candidate)
    markdown = render_getonboard_sync_markdown(package)

    assert "Jofré" not in fields.signature
    assert "Jofre" not in fields.signature
    assert "Jofre" not in markdown
    assert "Paredes" in markdown


def test_portal_draft_shows_the_candidates_own_skills() -> None:
    """The 'stack habitual' line filtered skills through a data-science whitelist."""
    candidate = Candidate.model_validate(nurse_profile_dict())

    text = build_permanent_profile_fields(candidate).experiencia_y_perfil

    assert "Ventilación Mecánica" in text


def test_imported_cv_keeps_its_own_skill_groups() -> None:
    """Every imported profile used to start with machine_learning and cloud groups."""
    text = (
        "Rocío Paredes Lagos\n"
        "ENFERMERA CLÍNICA\n"
        "HABILIDADES CLÍNICAS\n"
        "Ventilación Mecánica | Reanimación Cardiopulmonar | Registro Clínico\n"
    )

    skills = parse_cv_text(text, Path("cv.pdf")).data["skills"]

    assert "machine_learning" not in skills
    assert "cloud" not in skills
    assert any("Ventilación Mecánica" in group for group in skills.values())


def test_confirmed_skills_land_in_a_neutral_group() -> None:
    """Promoting a market skill used to file it under machine_learning."""
    from jobbot.profile.market import apply_confirmed_skills

    candidate = Candidate.model_validate(nurse_profile_dict())

    updated = apply_confirmed_skills(candidate, ["Triage Avanzado"])
    groups = updated.skills.as_dict()

    assert "Triage Avanzado" in updated.skills.all_skills()
    assert "Triage Avanzado" not in groups.get("machine_learning", [])


def test_torre_maps_an_opportunity_from_any_field(project_root: Path) -> None:
    """The mapper must not depend on the vocabulary of one profession."""
    import json

    from jobbot.adapters.torre.jobs import job_from_api_item

    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_search.json").read_text(encoding="utf-8")
    )
    clinical = job_from_api_item(payload["results"][1])
    newsroom = job_from_api_item(payload["results"][2])

    assert clinical.title == "Enfermera Clínica"
    assert "Cuidados intensivos" in clinical.skills
    assert newsroom.title == "Redactor de Contenidos"
    assert newsroom.description.startswith("Redactor de Contenidos")


def test_empleos_publicos_ficha_is_field_agnostic(project_root: Path) -> None:
    """A public-sector concurso parse must not assume a tech role vocabulary."""
    from jobbot.adapters.empleospublicos.jobs import job_from_ficha_html

    html = (project_root / "tests/fixtures/empleospublicos/aviso_ficha.html").read_text(
        encoding="utf-8"
    )
    job = job_from_ficha_html(
        html,
        url="https://www.empleospublicos.cl/pub/convocatorias/avisotrabajoficha.aspx?i=990001",
    )

    assert "Epidemiólogo" in job.title
    assert "vigilancia" in (job.description or "").casefold()
    assert job.ats_kind == "empleos_publicos"


def test_form_learning_reads_any_field_s_application(project_root: Path) -> None:
    """A form is read by its markup, so a clinic's form works like a studio's."""
    from jobbot.portals.form_learn import learn_form_html

    clinic = (
        "<html><body><form action='/postular'>"
        "<label for='email'>Correo</label><input type='email' id='email' name='email' required>"
        "<label for='cv'>Currículum</label><input type='file' id='cv' name='cv' required>"
        "<label for='reg'>N° de registro en la Superintendencia de Salud</label>"
        "<input type='text' id='reg' name='reg' required>"
        "</form></body></html>"
    )

    form = learn_form_html(clinic, url="https://clinica.example.cl/postular")
    labels = [field.label for field in form.fields]

    assert form.readable
    assert "N° de registro en la Superintendencia de Salud" in labels
    assert form.screening_questions() == ["N° de registro en la Superintendencia de Salud"]


def test_sso_buttons_are_identity_hosts_not_a_trade() -> None:
    """Google/LinkedIn on a clinic login are still SSO, not a field of work."""
    from jobbot.portals.sso import SsoProvider, detect_sso_providers

    html = (
        "<html><body>"
        "<button type='button'>Iniciar sesión con Google</button>"
        "<a href='https://www.linkedin.com/oauth/v2/authorization'>Continuar con LinkedIn</a>"
        "</body></html>"
    )
    found = detect_sso_providers(html)
    assert SsoProvider.GOOGLE in found
    assert SsoProvider.LINKEDIN in found


def test_the_cv_advisor_speaks_about_documents_not_about_a_trade() -> None:
    """Every suggestion must make sense for a nurse and for an engineer alike."""
    from jobbot.cv.advisor import advise

    for maker in (nurse_profile_dict, journalist_profile_dict):
        raw = maker()
        raw["experience"][0]["achievements"][0]["text"] = "Fui responsable de coordinar el equipo."
        candidate = Candidate.model_validate(raw)

        advice = advise(candidate, limit=5)

        blob = " ".join(f"{item.what} {item.why}" for item in advice).casefold()
        for word in ("python", "sql", "machine learning", "paciente", "noticia"):
            assert word not in blob


def test_recruiter_practices_are_extracted_without_a_field_s_words() -> None:
    from jobbot.recruiters.playbook import extract_practices

    html = (
        "<html><body><h1>Cómo leemos un CV</h1>"
        "<ul><li>Cuantifica el resultado: un número dice más que un adjetivo.</li>"
        "<li>Escribe la sigla y su nombre completo la primera vez.</li></ul>"
        "</body></html>"
    )

    practices = extract_practices(html)

    assert practices
    blob = " ".join(practice.text for practice in practices).casefold()
    for word in ("python", "enfermer", "periodis", "ingenier"):
        assert word not in blob


def test_career_page_reads_any_field_s_vacancy() -> None:
    """Career HTML is structure, so a clinic posting parses like a studio's."""
    from jobbot.jobs.career_page import job_from_career_html

    html = (
        "<html><body>"
        '<h1 data-ph-at-id="job-title">Enfermera Clínica</h1>'
        '<div data-ph-at-id="job-company">Clínica Cordillera</div>'
        '<div data-ph-at-id="job-description">'
        "<p>Requisitos: ventilación mecánica y registro clínico en ficha electrónica. "
        "Experiencia en fármacos vasoactivos.</p>"
        "</div>"
        '<div class="hide job-expired-view">the job you are trying to apply for '
        "has been filled.</div>"
        "</body></html>"
    )

    job = job_from_career_html(html, url="https://careers.clinic.example/job/1")
    assert job.title == "Enfermera Clínica"
    assert job.company == "Clínica Cordillera"
    assert "ventilación mecánica" in job.description.casefold()


def test_board_searches_come_from_each_profile_s_own_experience() -> None:
    """No default role: a nurse and a journalist search for their own work."""
    from jobbot.profile.search_queries import search_queries_for

    def searches(raw: dict[str, object]) -> str:
        return " | ".join(search_queries_for(Candidate.model_validate(raw)).queries)

    nurse = searches(nurse_profile_dict())
    journalist = searches(journalist_profile_dict())

    assert nurse and journalist
    assert nurse != journalist
    for blob in (nurse.casefold(), journalist.casefold()):
        assert "data" not in blob
        assert "python" not in blob


NURSE_CONDITIONS_JOB = (
    "Title: Enfermera Clínica\n"
    "Company: Clínica Cordillera\n"
    "Location: Valparaíso, Chile\n"
    "Requisitos excluyentes:\n"
    "- Registro clínico en ficha electrónica.\n"
    "- Curso de manejo de heridas avanzadas vigente.\n"
    "Deseable:\n"
    "- Diplomado en gestión del cuidado.\n"
    "Debe residir en Chile.\n"
    "Contrato indefinido, turnos rotativos.\n"
)


def test_posting_conditions_read_a_clinical_posting() -> None:
    """Mandatory/desirable and residency come from the wording, not from a field's terms."""
    from jobbot.jobs.conditions import ConditionKind, posting_conditions

    conditions = posting_conditions(parse_job_text(NURSE_CONDITIONS_JOB, job_id="J0620"))
    requirements = {
        c.detail: c.mandatory for c in conditions if c.kind is ConditionKind.REQUIREMENT
    }

    assert requirements["Registro clínico en ficha electrónica"] is True
    assert requirements["Diplomado en gestión del cuidado"] is False
    residency = [c for c in conditions if c.kind is ConditionKind.RESIDENCY]
    assert residency[0].countries == ("CL",)
    assert [c.contract for c in conditions if c.kind is ConditionKind.CONTRACT] == ["indefinite"]


def test_eligibility_judges_a_nurse_against_a_clinical_posting() -> None:
    """The nurse's own skills back an excluyente; what she never claimed becomes a question."""
    from jobbot.jobs.conditions import ConditionKind, posting_conditions
    from jobbot.jobs.eligibility import (
        ApplicationAnswers,
        VerdictStatus,
        assess_conditions,
        has_dealbreaker,
    )

    candidate = Candidate.model_validate(nurse_profile_dict())
    job = parse_job_text(NURSE_CONDITIONS_JOB, job_id="J0621")
    verdicts = assess_conditions(posting_conditions(job), candidate, ApplicationAnswers())
    mandatory = {
        v.condition.detail: v.status
        for v in verdicts
        if v.condition.kind is ConditionKind.REQUIREMENT and v.condition.mandatory
    }

    assert mandatory["Registro clínico en ficha electrónica"] is VerdictStatus.MEETS
    assert mandatory["Curso de manejo de heridas avanzadas vigente"] is VerdictStatus.ASK
    residency = [v for v in verdicts if v.condition.kind is ConditionKind.RESIDENCY]
    assert residency[0].status is VerdictStatus.MEETS
    assert not has_dealbreaker(verdicts)
