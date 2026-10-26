"""ATS apply plan tests."""

from jobbot.adapters.ats.apply import build_apply_plan, prefill_field_map
from jobbot.adapters.base import ApplyMethod
from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.job import JobPosting


def test_build_apply_plan_from_ats_url() -> None:
    cand = Candidate(
        personal=PersonalInfo(
            name="Leonardo Jofré",
            headline="DS",
            email="leo@example.com",
            linkedin="https://www.linkedin.com/in/leonardojofre",
        )
    )
    job = JobPosting(
        id="J1",
        title="DS",
        company="Acme",
        ats_url="https://boards.greenhouse.io/acme/jobs/1",
        ats_kind="greenhouse",
    )
    plan = build_apply_plan(cand, job)
    assert plan.method == ApplyMethod.EXTERNAL_ATS
    assert plan.ats_kind.value == "greenhouse"
    assert "prefill" in plan.message.casefold()
    fields = prefill_field_map(cand)
    assert fields["email"] == "leo@example.com"
    assert fields["first_name"] == "Leonardo"
    assert fields["last_name"] == "Jofré"
    assert "full_name" in fields


def test_unknown_ats_plan_does_not_claim_prefill() -> None:
    cand = Candidate(personal=PersonalInfo(name="Ana", headline="Editor"))
    job = JobPosting(
        id="J1",
        title="Editor",
        company="Northwind",
        ats_url="https://careers.example-corp.test/job/1",
        ats_kind="unknown",
    )
    plan = build_apply_plan(cand, job)
    assert plan.method == ApplyMethod.EXTERNAL_ATS
    assert "prefill" not in plan.message.casefold()
    assert "cheat sheet" in plan.message.casefold()


def test_four_part_name_keeps_both_surnames() -> None:
    """Regression: the first space used to swallow the second given name."""
    from jobbot.adapters.ats.apply import prefill_sheet_fields, split_personal_name

    given, surnames = split_personal_name("Ada María López Soto")
    assert given == "Ada María"
    assert surnames == "López Soto"
    cand = Candidate(
        personal=PersonalInfo(
            name="Ada María López Soto",
            headline="Editor",
            email="ada@example.com",
            phone="+56 9 1234 5678",
        )
    )
    fields = prefill_field_map(cand)
    assert fields["first_name"] == "Ada María"
    assert fields["last_name"] == "López Soto"
    sheet = prefill_sheet_fields(cand)
    assert "email" not in sheet
    assert "phone" not in sheet
    assert "ada@example.com" not in sheet.values()
    assert sheet["last_name"] == "López Soto"


def test_lever_adapter_detects() -> None:
    from jobbot.adapters.ats.lever import LeverAdapter

    job = JobPosting(
        id="J1",
        title="DS",
        company="Acme",
        ats_url="https://jobs.lever.co/acme/abc",
    )
    adapter = LeverAdapter()
    assert adapter.detect_method(job) == ApplyMethod.EXTERNAL_ATS


def test_getonboard_adapter_prefill_includes_spanish_hint() -> None:
    from jobbot.adapters.ats.getonboard import GetOnBoardAdapter

    cand = Candidate(personal=PersonalInfo(name="Ana", headline="DS"))
    job = JobPosting(
        id="J1",
        title="DS",
        company="Co",
        ats_url="https://www.getonbrd.com/jobs/x",
    )
    result = GetOnBoardAdapter().prefill(cand, job, package=_DummyPackage())
    assert any("español" in item.casefold() for item in result.needs_review)


def test_greenhouse_adapter_detects() -> None:
    from jobbot.adapters.ats.greenhouse import GreenhouseAdapter

    cand = Candidate(personal=PersonalInfo(name="Ana", headline="DS"))
    job = JobPosting(
        id="J1",
        title="DS",
        company="Acme",
        ats_url="https://boards.greenhouse.io/acme/jobs/1",
    )
    adapter = GreenhouseAdapter()
    assert adapter.detect_method(job) == ApplyMethod.EXTERNAL_ATS
    result = adapter.prefill(cand, job, package=_DummyPackage())
    assert "full_name" in result.filled or "email" in result.filled or result.filled


class _DummyPackage:
    @property
    def directory(self):  # noqa: ANN201
        from pathlib import Path

        return Path(".")

    @property
    def cv_pdf(self):  # noqa: ANN201
        return None


def test_resolve_honors_stored_kind_on_custom_career_domain() -> None:
    """Host rules miss careers.neuralworks.cl; stored sniff kind must still drive the plan."""
    from jobbot.adapters.ats.apply import resolve_ats_url
    from jobbot.portals.detect import AtsKind

    job = JobPosting(
        id="J9",
        title="Applied Scientist",
        company="NeuralWorks",
        ats_url="https://careers.neuralworks.cl/jobs/568945-applied-scientist",
        ats_kind="teamtailor",
    )
    url, kind = resolve_ats_url(job)
    assert url == job.ats_url
    assert kind == AtsKind.TEAMTAILOR
    plan = build_apply_plan(
        Candidate(personal=PersonalInfo(name="Ana", headline="DS")),
        job,
    )
    assert plan.ats_kind == AtsKind.TEAMTAILOR
    assert plan.method == ApplyMethod.EXTERNAL_ATS


def test_linkedin_message_plan_opens_the_post_and_does_not_prefill() -> None:
    from jobbot.adapters.ats.apply import resolve_ats_url
    from jobbot.portals.detect import AtsKind

    cand = Candidate(personal=PersonalInfo(name="Ana", headline="Editor"))
    job = JobPosting(
        id="J1",
        title="Editor de Contenidos",
        company="Unknown company",
        url="https://www.linkedin.com/posts/recruiter_activity-7511877806738411520-MJ5G",
        description="Si te interesa, mandame un mensaje indicando que te interesa este puesto.",
    )
    url, kind = resolve_ats_url(job)
    assert url == job.url
    assert kind == AtsKind.LINKEDIN
    plan = build_apply_plan(cand, job)
    assert plan.method == ApplyMethod.LINKEDIN_MESSAGE
    assert "never sends" in plan.message.casefold()
    assert plan.fields_to_fill == []


def test_linkedin_permalink_without_a_message_ask_is_not_an_ats() -> None:
    from jobbot.adapters.ats.apply import resolve_ats_url
    from jobbot.portals.detect import AtsKind

    job = JobPosting(
        id="J1",
        title="Machine Learning",
        company="NORTHWIND DATA",
        url="https://www.linkedin.com/posts/recruiter_activity-1-xxxx",
        description="Buscamos talento. El link de postulación está en el primer comentario.",
    )
    url, kind = resolve_ats_url(job)
    assert url is None
    assert kind == AtsKind.UNKNOWN
    plan = build_apply_plan(
        Candidate(personal=PersonalInfo(name="Ana", headline="Editor")),
        job,
    )
    assert plan.method == ApplyMethod.UNKNOWN
    assert plan.ats_url == ""
