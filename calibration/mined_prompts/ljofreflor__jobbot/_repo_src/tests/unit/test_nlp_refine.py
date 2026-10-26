"""Cumulative profile refine tests — previous text is computational capital."""

from jobbot.adapters.getonboard.draft import PermanentProfileFields
from jobbot.models.candidate import Candidate
from jobbot.nlp.refine import CumulativeProfileRefiner, refine_permanent_profile
from tests.fixtures.profile import sample_profile_dict


def _previous_obsolete() -> PermanentProfileFields:
    return PermanentProfileFields(
        experiencia_y_perfil=(
            "Me he dedicado a Matlab, VTK y Mayavi en Ceamos www.ceamos.cl.\n\n"
            "También trabajé modelos en Python en Mercado Libre con share of wallet."
        ),
        formacion_academica="Magíster en Estadística — PUC.",
        headline="Old headline",
        skills=["Matlab"],
    )


def test_cumulative_drops_obsolete_keeps_supported() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    result = CumulativeProfileRefiner().refine_with_stats(candidate, _previous_obsolete())
    assert result.mode == "cumulative"
    assert result.dropped_paragraphs >= 1
    text = result.fields.experiencia_y_perfil.casefold()
    assert "ceamos" not in text
    assert "matlab" not in text
    assert "mercado libre" in text or "data scientist" in text


def test_a_truncated_previous_paragraph_is_upgraded_not_preserved() -> None:
    """Accumulating must not mean keeping a mutilated version of the same text.

    A stored paragraph that is a prefix of the freshly built one lost information
    (an old length cap cut it mid-sentence), so the complete text has to win.
    """
    candidate = Candidate.model_validate(sample_profile_dict())
    complete = (candidate.summary or "").strip()
    assert complete, "the sample profile needs a summary for this test"
    truncated = complete[: int(len(complete) * 0.6)].rsplit(" ", 1)[0]

    previous = PermanentProfileFields(
        experiencia_y_perfil=truncated,
        formacion_academica="Magíster en Estadística — PUC.",
        headline="Senior Data Scientist",
        skills=["Python"],
    )
    result = CumulativeProfileRefiner().refine_with_stats(candidate, previous)

    assert complete in result.fields.experiencia_y_perfil
    assert truncated not in result.fields.experiencia_y_perfil.replace(complete, "")


def test_cold_start_when_no_previous() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    result = refine_permanent_profile(candidate, None)
    assert result.mode == "cold"
    assert result.fields.experiencia_y_perfil


def test_refine_does_not_invent_kubernetes() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    result = refine_permanent_profile(candidate, _previous_obsolete())
    assert "kubernetes" not in result.fields.experiencia_y_perfil.casefold()


def test_a_paragraph_listing_the_candidates_own_tools_survives() -> None:
    """The anchors held employers and titles only, so a stack line looked unbacked."""
    candidate = Candidate.model_validate(sample_profile_dict())
    tools = list(candidate.skills.all_skills())[:4]
    stack = "Stack habitual: " + ", ".join(tools) + "."
    previous = PermanentProfileFields(
        experiencia_y_perfil=stack,
        formacion_academica="Formación previa.",
        headline=candidate.personal.headline or "",
        skills=tools,
    )

    result = CumulativeProfileRefiner().refine_with_stats(candidate, previous)

    assert stack in result.fields.experiencia_y_perfil
    assert result.dropped_paragraphs == 0


def test_refining_twice_does_not_erode_the_text() -> None:
    """Each run fed its own output back in: cold added a paragraph, the next run cut it."""
    candidate = Candidate.model_validate(sample_profile_dict())
    first = refine_permanent_profile(candidate, None)
    second = refine_permanent_profile(candidate, first.fields)
    third = refine_permanent_profile(candidate, second.fields)

    assert second.fields.experiencia_y_perfil == third.fields.experiencia_y_perfil
    assert second.dropped_paragraphs == 0
    assert third.dropped_paragraphs == 0
