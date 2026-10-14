"""The repository boundary, as a check rather than a convention.

The benchmark holds the womblex-collection ground truth and runs the suites
that score against it; this repository holds the code under test and receives
the reports. The failure that made this worth enforcing was not a missing rule
but a duplicated suite: two copies of the same scorer, one here and one there,
both writing the same file under ``docs/accuracy/``, so which numbers got
published depended on which was run last. They then drifted 81, 54 and 191
lines apart without either being wrong.

So the invariant is about *producers*, not about fixtures. A unit test reading
a synthetic document out of ``fixtures/synthetic/`` is fine — that set is
committed precisely so a bare clone can run the suite (see
THIRD_PARTY_DATA.md). Writing an accuracy report from here is not.

Limitation: this reads source text, so it catches a test that names the
directory, which is how you reach it in practice. A test that assembled the
path indirectly would slip past.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
DOCS_ACCURACY = TESTS_DIR.parent / "docs" / "accuracy"

#: Ways a test could name the accuracy docs directory: as a path fragment, or
#: assembled from its two segments by ``/`` or by a joining call.
_NAMES_ACCURACY_DOCS = re.compile(
    r"""docs/accuracy
      | ["']docs["']\s*/\s*["']accuracy["']
      | ["']docs["']\s*,\s*["']accuracy["']
    """,
    re.VERBOSE,
)


def _test_sources() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("*.py") if p.name != Path(__file__).name)


def test_no_library_test_writes_an_accuracy_report():
    """Accuracy reports have exactly one producing suite, and it is not here.

    A hit means a suite has been added or restored on this side of the
    boundary. Move it to the benchmark's ``accuracy/`` directory and let it
    publish through ``_paths.WOMBLEX_DOCS``, rather than adding an exemption.
    """
    offenders = [
        p.name for p in _test_sources()
        if _NAMES_ACCURACY_DOCS.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "these tests name the accuracy docs directory, so they can publish a "
        f"report this repository does not own: {offenders}. The suites that "
        "score against womblex-collection ground truth live in the benchmark."
    )


def test_the_accuracy_reports_are_still_published_here():
    """The other half of the boundary: the reports do land in this repository.

    Guards against reading the check above as "no accuracy docs" rather than
    "no accuracy docs *written from here*" — if the directory ever empties,
    the first check would pass for the wrong reason.
    """
    assert DOCS_ACCURACY.is_dir(), f"{DOCS_ACCURACY} is missing"
    reports = sorted(p.name for p in DOCS_ACCURACY.glob("*.md"))
    assert reports, "no accuracy reports are published here"
