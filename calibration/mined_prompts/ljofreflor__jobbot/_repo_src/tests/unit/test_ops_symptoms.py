"""Latent requirements (symptoms) — redacted capture → System 1 compression."""

from __future__ import annotations

from pathlib import Path

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.ops.compile import PROMOTION_STEPS
from jobbot.ops.symptoms import (
    get_symptom,
    issue_body,
    list_symptoms,
    mark_symptom_status,
    note_symptom,
    promote_plan,
    sanitize_symptom_text,
    symptom_fingerprint,
)


def _session(tmp_path: Path):
    config = JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(
            profile=Path("data/profile.yaml"),
            database=Path("data/jobbot.sqlite"),
            output=Path("output"),
        ),
    )
    engine = make_engine(config.database_path)
    return make_session_factory(engine)(), config


def test_sanitize_strips_email_phone_and_home() -> None:
    raw = (
        "match fails for ana@empresa-real.cl call +56 9 1234 5678 "
        "see /Users/ljofre/cv.tex and OPENAI_API_KEY=sk-abcdefghijklmnop"
    )
    clean = sanitize_symptom_text(raw)
    assert "@" not in clean or "<email>" in clean
    assert "1234" not in clean
    assert "/Users/" not in clean
    assert "sk-abcdef" not in clean


def test_note_increments_sightings_on_same_fingerprint(tmp_path: Path) -> None:
    session, config = _session(tmp_path)
    a = note_symptom(
        session,
        intent="internship posts score 100% against senior profile",
        area="matching",
        rule_hypothesis="seniority must gate perfect skill overlap",
        output_dir=config.output_dir,
    )
    b = note_symptom(
        session,
        intent="internship posts score 100% against senior profile",
        area="matching",
        rule_hypothesis="seniority must gate perfect skill overlap",
        output_dir=config.output_dir,
    )
    assert a.id == b.id
    assert b.sightings == 2
    assert a.fingerprint == b.fingerprint
    mirror = config.output_dir / "ops" / "symptoms" / f"{a.id}.json"
    assert mirror.is_file()


def test_fingerprint_stable_across_noise() -> None:
    a = symptom_fingerprint(
        area="matching",
        intent="job J0016 internship scores 100",
        rule_hypothesis="gate by seniority",
    )
    b = symptom_fingerprint(
        area="matching",
        intent="job J0099 internship scores 100",
        rule_hypothesis="gate by seniority",
    )
    assert a == b


def test_promote_plan_includes_promotion_steps(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    record = note_symptom(
        session,
        intent="cover letter still drafted only in Cursor chat",
        area="nlp",
        rule_hypothesis="prose reconstitutes when prepare has no grounded draft path",
    )
    plan = promote_plan(record)
    assert record.id in plan
    assert "conditions of possibility" in plan.casefold()
    assert "wish-list" in plan or "requirement" in plan.casefold()
    assert "capture_symptom_redacted_locally" in PROMOTION_STEPS
    for step in PROMOTION_STEPS:
        assert step in plan


def test_issue_body_has_no_raw_secret(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    record = note_symptom(
        session,
        intent="sync fails with token=supersecret and mail me@corp.cl",
        area="ops",
    )
    body = issue_body(record)
    assert "supersecret" not in body
    assert "me@corp.cl" not in body
    assert "<email>" in record.intent or "email" in record.intent.casefold()


def test_triage_resolved_sets_feature_path(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    record = note_symptom(session, intent="detect ashby forms", area="portals")
    updated = mark_symptom_status(
        session,
        record.id,
        "resolved",
        feature_path="src/jobbot/adapters/ats/ashby.py",
    )
    assert updated is not None
    assert updated.status == "resolved"
    assert updated.feature_path is not None
    assert "ashby" in updated.feature_path
    assert get_symptom(session, record.id) is not None
    assert list_symptoms(session, status="resolved")
