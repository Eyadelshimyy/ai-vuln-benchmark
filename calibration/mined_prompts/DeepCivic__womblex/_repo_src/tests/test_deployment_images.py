"""The deployment-image audit still describes the compose files it audits.

`docs/deployment-images.md` records, per compose service, where its image comes
from. An audit is only worth having if it cannot go stale, so this reads the
document's tables back and holds them against the files themselves: the counts
it declares, a verdict on every service, and a `Kind`/`Reference` that still
says what compose says.

The failure this exists to catch is a service added later that builds from
source with nobody having decided it should — which looks like nothing at all
until a deployment cannot name what it is running.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = REPO_ROOT / "docs" / "deployment-images.md"

#: The audit's short names for the compose files, as its `File` column spells
#: them. The document is read by people as well as by this test, so it says
#: "base" rather than repeating the filename on all thirteen rows.
COMPOSE_FILES = {
    "base": REPO_ROOT / "docker-compose.yml",
    "override": REPO_ROOT / "docker-compose.local.override.yml",
}

_EM_DASH = "—"


def _markdown_rows(text: str) -> list[list[str]]:
    """Every pipe-table body row in *text*, as stripped cells.

    Header and separator rows are dropped: a separator is all dashes and colons,
    and the header immediately precedes one. Good enough for a document whose
    tables this repository writes, and it keeps the audit a document rather than
    a data file with prose around it.
    """
    rows: list[list[str]] = []
    lines = [line.strip() for line in text.splitlines()]
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        is_separator = all(set(c) <= set("-: ") and c for c in cells)
        next_is_separator = i + 1 < len(lines) and lines[i + 1].startswith("|") and all(
            set(c.strip()) <= set("-: ") and c.strip()
            for c in lines[i + 1].strip().strip("|").split("|")
        )
        if is_separator or next_is_separator:
            continue
        rows.append(cells)
    return rows


@pytest.fixture(scope="module")
def audit_text() -> str:
    return AUDIT.read_text()


@pytest.fixture(scope="module")
def declared(audit_text: str) -> dict[str, int]:
    """The audit's "Declared totals" table, as label -> count."""
    return {
        cells[0]: int(cells[1])
        for cells in _markdown_rows(audit_text)
        if len(cells) == 2 and cells[1].isdigit()
    }


@pytest.fixture(scope="module")
def enumerated(audit_text: str) -> list[dict[str, str]]:
    """The audit's per-service table, one dict per row."""
    columns = ("service", "file", "kind", "reference", "profile", "verdict")
    return [
        dict(zip(columns, cells, strict=True))
        for cells in _markdown_rows(audit_text)
        if len(cells) == len(columns)
    ]


@pytest.fixture(scope="module")
def services() -> dict[str, dict]:
    """Every compose service keyed by (file label, service name)."""
    out = {}
    for label, path in COMPOSE_FILES.items():
        for name, body in yaml.safe_load(path.read_text())["services"].items():
            out[(label, name)] = body or {}
    return out


#: What marks an image reference as this project's own rather than a third
#: party's. The published references are `${WOMBLEX_*_IMAGE:-ghcr.io/...}`, so
#: either half of that carries the name.
_OURS = "womblex"


def _kind(body: dict) -> str:
    """`build`, `published` (an image of this project), `image` (third party).

    Published and third-party are separated because the audit asks a different
    question of each: a third-party reference is audited for whether its tag
    moves, while one of ours is audited for whether a deployment can pin it.
    """
    if "build" in body:
        return "build"
    if "image" not in body:
        return "settings"
    return "published" if _OURS in body["image"] else "image"


def _reference(body: dict) -> str:
    if "build" in body:
        return body["build"]["dockerfile"]
    return body.get("image", _EM_DASH)


class TestTheAuditIsComplete:
    """Every service is enumerated, and nothing is enumerated that does not exist."""

    def test_the_declared_counts_are_the_real_counts(
        self, declared: dict[str, int], services: dict,
    ) -> None:
        """The criterion the whole audit rests on: a missed service is visible.

        An enumeration checked only against itself would pass while silently
        omitting a service, which is the exact failure being guarded.
        """
        base = {k: v for k, v in services.items() if k[0] == "base"}
        override = {k: v for k, v in services.items() if k[0] == "override"}
        kinds = [_kind(v) for v in base.values()]

        assert declared["Compose files"] == len(COMPOSE_FILES)
        assert declared["Services in `docker-compose.yml`"] == len(base)
        assert declared["Services in `docker-compose.local.override.yml`"] == len(override)
        assert declared["Of the base file: build from source"] == kinds.count("build")
        assert declared["Of the base file: reference a third-party image"] == kinds.count(
            "image",
        )
        assert declared[
            "Of the base file: reference an image of this project"
        ] == kinds.count("published")
        assert declared["Of the override: build from source"] == [
            _kind(v) for k, v in services.items() if k[0] == "override"
        ].count("build")

    def test_every_service_is_enumerated_exactly_once(
        self, enumerated: list[dict[str, str]], services: dict,
    ) -> None:
        listed = [(row["file"], row["service"].strip("`")) for row in enumerated]
        assert sorted(listed) == sorted(services)
        assert len(listed) == len(set(listed)), "a service is enumerated twice"

    def test_the_enumeration_matches_the_declared_total(
        self, enumerated: list[dict[str, str]], declared: dict[str, int],
    ) -> None:
        expected = (
            declared["Services in `docker-compose.yml`"]
            + declared["Services in `docker-compose.local.override.yml`"]
        )
        assert len(enumerated) == expected


class TestEveryServiceHasAVerdict:
    """No service is left without a recorded decision — the R5 criterion."""

    def test_no_verdict_is_empty(self, enumerated: list[dict[str, str]]) -> None:
        missing = [r["service"] for r in enumerated if not r["verdict"].strip(f" {_EM_DASH}")]
        assert not missing, f"services with no recorded decision: {missing}"

    def test_a_published_service_is_pinnable_by_a_declared_value(
        self, enumerated: list[dict[str, str]], services: dict,
    ) -> None:
        """The switch between published and built is one value, not a file edit.

        A bare `image: ghcr.io/...` would satisfy "references an image" while
        leaving a deployment no way to name a digest, which is the criterion
        that actually matters.
        """
        for row in (r for r in enumerated if r["kind"] == "published"):
            reference = _reference(services[(row["file"], row["service"].strip("`"))])
            assert reference.startswith("${WOMBLEX_") and ":-" in reference, (
                f"{row['service']} names a published image but not through an "
                f"overridable value: {reference!r}"
            )

    def test_a_build_from_source_service_names_where_its_image_comes_from(
        self, enumerated: list[dict[str, str]],
    ) -> None:
        """A verdict that does not say "published" must say why it is built locally.

        Both are acceptable answers; a verdict that says neither is the
        unaccounted-for service the requirement is about.
        """
        for row in (r for r in enumerated if r["kind"] == "build"):
            verdict = row["verdict"].lower()
            assert "published" in verdict or "built locally" in verdict, (
                f"{row['service']} builds from source but its verdict neither "
                f"publishes it nor states a reason to build it: {row['verdict']!r}"
            )

    def test_every_third_party_image_records_whether_its_tag_is_pinned(
        self, enumerated: list[dict[str, str]],
    ) -> None:
        """A moving tag is the same unpinnability as a local build, elsewhere."""
        for row in (r for r in enumerated if r["kind"] == "image"):
            assert "pinned" in row["verdict"].lower(), (
                f"{row['service']} references an image but its verdict does not "
                f"say whether the tag is pinned: {row['verdict']!r}"
            )

    def test_an_unpinned_reference_is_not_recorded_as_pinned(
        self, enumerated: list[dict[str, str]], services: dict,
    ) -> None:
        """The pinning verdict tracks the reference rather than being asserted.

        `image: minio/minio` carries no tag, so it resolves to whatever the
        registry serves; a verdict calling that pinned would be the audit
        stating something untrue about a file in the same repository.
        """
        for row in (r for r in enumerated if r["kind"] == "image"):
            reference = _reference(services[(row["file"], row["service"].strip("`"))])
            tagged = ":" in reference.rsplit("/", 1)[-1]
            says_unpinned = "unpinned" in row["verdict"].lower()
            assert tagged is not says_unpinned, (
                f"{row['service']} references {reference!r} but its verdict says "
                f"{'unpinned' if says_unpinned else 'pinned'}"
            )


class TestTheAuditStillDescribesTheFiles:
    """A verdict on a service whose shape has moved under it is worse than none."""

    def test_each_row_names_what_compose_declares(
        self, enumerated: list[dict[str, str]], services: dict,
    ) -> None:
        for row in enumerated:
            body = services[(row["file"], row["service"].strip("`"))]
            assert row["kind"] == _kind(body), (
                f"{row['service']} ({row['file']}) is audited as {row['kind']!r} "
                f"but compose declares {_kind(body)!r}"
            )
            assert row["reference"].strip("`") == _reference(body)

    def test_a_profile_is_recorded_rather_than_exempting_the_service(
        self, enumerated: list[dict[str, str]], services: dict,
    ) -> None:
        """Behind a profile is not an exemption — it is a column."""
        for row in enumerated:
            body = services[(row["file"], row["service"].strip("`"))]
            profiles = body.get("profiles") or []
            recorded = row["profile"].strip(f" `{_EM_DASH}")
            assert recorded == (profiles[0] if profiles else ""), (
                f"{row['service']} declares profiles {profiles} but the audit "
                f"records {row['profile']!r}"
            )
