"""Get on Board sync package tests."""

from jobbot.adapters.getonboard.package import (
    PROFILE_EDIT_URL,
    build_getonboard_sync_package,
    render_getonboard_sync_markdown,
)
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict


def test_sync_package_from_profile_facts_only() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    package = build_getonboard_sync_package(candidate)
    assert package.headline
    assert "Data Scientist" in package.headline or package.summary
    assert len(package.skills) <= 10
    assert package.profile_edit_url == PROFILE_EDIT_URL
    md = render_getonboard_sync_markdown(package)
    assert "Editar perfil" in md or "webpros/edit" in md
    assert "Tus CVs" in md
    assert "Kubernetes" not in md  # not inventing
