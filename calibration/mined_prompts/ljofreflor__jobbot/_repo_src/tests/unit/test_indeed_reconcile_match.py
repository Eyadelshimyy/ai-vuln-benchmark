"""Matching helpers for Indeed reconcile (full mirror)."""

from jobbot.adapters.indeed.reconcile import match_education, match_experience
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict


def test_match_experience_prefers_company_and_title() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    exp = candidate.experience[0]
    assert match_experience(exp.title, exp.company, exp) >= 0.9
    assert match_experience("desarrollo", "Centro de modelamiento matemático", exp) < 0.5


def test_match_education_incomplete_engineering_row() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    # sample has Universidad de Chile magíster — use real UTEM-like local via mutate
    edu = candidate.education[0]
    score = match_education(edu.degree, "", edu)
    assert score >= 0.5
