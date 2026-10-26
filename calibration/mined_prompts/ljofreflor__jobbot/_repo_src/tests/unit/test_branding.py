"""The public mark is one closing line, and it does not stack."""

from jobbot.adapters.getonboard.draft import EXPERIENCE_MAX, build_permanent_profile_fields
from jobbot.adapters.indeed.package import build_indeed_sync_package
from jobbot.branding import MARK, stamp_description, strip_mark
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict


def test_stamp_once_and_strip() -> None:
    once = stamp_description("Data scientist.", max_len=200)
    assert once.endswith(MARK)
    assert stamp_description(once).count(MARK) == 1
    assert strip_mark(once) == "Data scientist."


def test_blank_stays_blank() -> None:
    assert stamp_description("  ") == ""


def test_stamp_respects_limit() -> None:
    text = stamp_description("palabra " * 500, max_len=80)
    assert len(text) <= 80
    assert text.endswith(MARK)


def test_published_descriptions_carry_the_mark() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    fields = build_permanent_profile_fields(candidate)
    package = build_indeed_sync_package(candidate)
    assert fields.experiencia_y_perfil.endswith(MARK)
    assert len(fields.experiencia_y_perfil) <= EXPERIENCE_MAX
    assert package.summary.endswith(MARK)
    assert package.headline.endswith(MARK) is False
