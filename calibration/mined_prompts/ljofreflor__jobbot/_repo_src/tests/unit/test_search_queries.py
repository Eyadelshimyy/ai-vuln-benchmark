"""Search queries come from what the candidate has done, not from a headline or a default.

A headline can be a degree ('Doctora en Ciencias (en curso)'), and a literal default
query assumes one career for everybody. The competencies the experience backs, the
phrases the achievements repeat and the titles actually held are the searches.
"""

from __future__ import annotations

from jobbot.jobs.normalization import fold_text
from jobbot.models.candidate import Candidate
from jobbot.profile.search_queries import (
    QueryOrigin,
    derive_search_queries,
    search_queries_for,
)
from tests.fixtures.profile import (
    nurse_profile_dict,
    public_health_profile_dict,
    sample_profile_dict,
)


def _texts(candidate: Candidate) -> list[str]:
    return [fold_text(query.text) for query in derive_search_queries(candidate)]


def test_queries_come_from_competencies_the_experience_backs() -> None:
    texts = _texts(Candidate.model_validate(public_health_profile_dict()))

    assert "vigilancia epidemiologica" in texts
    assert "investigacion de brotes" in texts


def test_a_degree_headline_is_not_a_query() -> None:
    texts = " | ".join(_texts(Candidate.model_validate(public_health_profile_dict())))

    assert "doctora" not in texts
    assert "magister" not in texts


def test_titles_actually_held_are_queries() -> None:
    queries = derive_search_queries(Candidate.model_validate(public_health_profile_dict()))

    titles = {fold_text(q.text) for q in queries if q.origin is QueryOrigin.TITLE}
    assert "referente de vigilancia de zoonosis" in titles or "consultora internacional" in titles


def test_a_skill_no_experience_backs_is_not_a_query() -> None:
    """'R' and 'QGIS' are listed, but no role mentions them: searching them is a guess."""
    texts = _texts(Candidate.model_validate(public_health_profile_dict()))

    assert "r" not in texts
    assert "qgis" not in texts


def test_every_query_cites_the_experiences_that_back_it() -> None:
    queries = derive_search_queries(Candidate.model_validate(public_health_profile_dict()))

    assert queries
    assert all(query.evidence for query in queries)


def test_queries_mix_competencies_and_titles() -> None:
    queries = derive_search_queries(Candidate.model_validate(public_health_profile_dict()))

    origins = {query.origin for query in queries}
    assert QueryOrigin.COMPETENCY in origins
    assert QueryOrigin.TITLE in origins


def test_no_query_is_repeated() -> None:
    texts = _texts(Candidate.model_validate(public_health_profile_dict()))

    assert len(texts) == len(set(texts))


def test_limit_is_respected() -> None:
    candidate = Candidate.model_validate(public_health_profile_dict())

    assert len(derive_search_queries(candidate, limit=3)) == 3


def test_a_profile_without_skills_still_gets_queries_from_its_achievements() -> None:
    """A PDF import can come back with zero skills; the achievements still speak."""
    raw = public_health_profile_dict()
    raw["skills"] = {}
    queries = derive_search_queries(Candidate.model_validate(raw))

    origins = {query.origin for query in queries}
    texts = " | ".join(fold_text(query.text) for query in queries)
    assert QueryOrigin.RECURRING in origins
    assert "vigilancia epidemiologica" in texts or "investigacion de brotes" in texts


def _role(exp_id: str, title: str, start: str, end: str, text: str) -> dict[str, object]:
    return {
        "id": exp_id,
        "company": "Institución Neutra",
        "title": title,
        "start_date": start,
        "end_date": end,
        "achievements": [{"id": f"{exp_id}-a", "text": text}],
    }


def test_a_recent_title_comes_before_an_old_one_held_often() -> None:
    """Regression: a teaching title held four times a decade ago led the searches."""
    raw = public_health_profile_dict()
    raw["experience"] = [
        _role("old-1", "Ayudante Docente", "2010-01", "2010-12", "Docencia de prácticos."),
        _role("old-2", "Ayudante Docente", "2011-01", "2011-12", "Docencia de prácticos."),
        _role("old-3", "Ayudante Docente", "2012-01", "2012-12", "Docencia de prácticos."),
        _role("new", "Consultora Internacional", "2024-01", "2025-06", "Vigilancia de brotes."),
    ]

    titles = [
        q.text
        for q in derive_search_queries(Candidate.model_validate(raw))
        if q.origin is QueryOrigin.TITLE
    ]

    assert titles[0] == "Consultora Internacional"


def test_a_phrase_two_roles_share_by_accident_is_not_a_query() -> None:
    """Regression: 'red pública, privada' in two achievements became a search."""
    raw = public_health_profile_dict()
    raw["skills"] = {}
    raw["experience"] = [
        _role(
            "a",
            "Referente Territorial",
            "2021-01",
            "2022-12",
            "Vigilancia epidemiológica en la red pública privada de laboratorios.",
        ),
        _role(
            "b",
            "Apoyo Técnico",
            "2020-01",
            "2020-12",
            "Vigilancia epidemiológica y gestión de la red pública privada.",
        ),
    ]

    texts = _texts(Candidate.model_validate(raw))

    assert "vigilancia epidemiologica" in texts
    assert "publica privada" not in texts


def test_a_data_profile_gets_data_queries_and_nobody_else_does() -> None:
    nurse = " | ".join(_texts(Candidate.model_validate(nurse_profile_dict())))
    data = " | ".join(_texts(Candidate.model_validate(sample_profile_dict())))

    assert "data scien" not in nurse
    assert "ventilacion mecanica" in nurse
    assert data


def test_saved_queries_win_over_derived_ones() -> None:
    raw = public_health_profile_dict()
    raw["search_queries"] = ["epidemiólogo", "salud pública remoto"]

    chosen = search_queries_for(Candidate.model_validate(raw))

    assert chosen.saved
    assert chosen.queries == ["epidemiólogo", "salud pública remoto"]


def test_without_saved_queries_the_derived_set_is_used() -> None:
    chosen = search_queries_for(Candidate.model_validate(public_health_profile_dict()))

    assert not chosen.saved
    assert chosen.queries


def test_an_empty_profile_has_no_queries() -> None:
    candidate = Candidate.model_validate({"personal": {"name": "Ana", "headline": "Ana"}})

    assert derive_search_queries(candidate) == []
    assert search_queries_for(candidate).queries == []


def test_blank_saved_queries_are_ignored() -> None:
    raw = public_health_profile_dict()
    raw["search_queries"] = ["  ", "epidemiólogo", "epidemiólogo"]

    assert Candidate.model_validate(raw).search_queries == ["epidemiólogo"]
