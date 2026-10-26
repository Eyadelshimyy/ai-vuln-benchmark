"""Market feedback suggestion tests — no invention, no deletions."""

from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.job import JobPosting
from jobbot.models.skill import SkillGroups
from jobbot.profile.market import apply_confirmed_skills, suggest_from_market


def _candidate() -> Candidate:
    return Candidate(
        personal=PersonalInfo(name="Ana", headline="Data Scientist"),
        summary="Builds ML models in Python.",
        skills=SkillGroups(programming=["Python", "SQL"], machine_learning=["XGBoost"]),
    )


def test_suggest_marks_present_and_asks_gaps() -> None:
    jobs = [
        JobPosting(
            id="J1",
            title="DS",
            company="A",
            description="Need Python, SQL, machine learning, causal inference, and dbt experience.",
        ),
        JobPosting(
            id="J2",
            title="ML",
            company="B",
            description="Python SQL machine learning causal inference pytorch dbt required.",
        ),
    ]
    suggestion = suggest_from_market(_candidate(), jobs, min_count=2)
    assert "python" in suggestion.present
    assert "sql" in suggestion.present
    assert "machine learning" in suggestion.present
    gap_terms = {g.term for g in suggestion.missing_suspected}
    assert "dbt" in gap_terms
    assert "machine learning" not in gap_terms
    # Does not auto-add
    assert all(g.term not in _candidate().skills.all_skills() for g in suggestion.missing_suspected)


def test_apply_confirmed_skills_does_not_delete() -> None:
    cand = _candidate()
    updated = apply_confirmed_skills(cand, ["causal inference", "dbt"])
    skills = updated.skills.all_skills()
    assert "Python" in skills
    assert "SQL" in skills
    assert "XGBoost" in skills
    assert "causal inference" in skills
    assert "dbt" in skills


def test_merge_confirmed_skills_into_raw_preserves_other_keys() -> None:
    from jobbot.profile.market import merge_confirmed_skills_into_raw

    raw = {
        "personal": {"name": "Ana", "headline": "DS"},
        "summary": "Keep me",
        "skills": {"programming": ["Python"], "machine_learning": ["XGBoost"]},
        "experience": [{"id": "e1", "company": "X", "title": "Y", "start_date": "2020-01"}],
    }
    merged = merge_confirmed_skills_into_raw(raw, ["dbt"])
    assert merged["summary"] == "Keep me"
    assert merged["experience"][0]["id"] == "e1"
    assert "Python" in merged["skills"]["programming"]
    # A confirmed skill lands in a neutral group: JobBot cannot know which group
    # it belongs to for a candidate whose field it does not know.
    assert "dbt" in merged["skills"]["other"]
    assert "XGBoost" in merged["skills"]["machine_learning"]
