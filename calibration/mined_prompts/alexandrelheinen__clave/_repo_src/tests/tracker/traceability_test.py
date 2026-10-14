"""The two criteria nothing else guards.



Both were satisfied by the code and by review, and neither was named by a test,
which the spec's own traceability rule forbids: an id has to be greppable in
both directions or the mapping rots the first time somebody changes the thing
it describes.
"""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BELT_FRAME_DOCUMENTS = (
    "docs/perception-contract.md",
    "crates/clave-decision/src/pose.rs",
    "configs/runtime/sitl.yml",
)
"""The three places that described a belt frame the code did not use."""


def test_no_document_describes_a_belt_frame_the_code_does_not_use() -> None:
    """No document describes a belt frame the code does not use.

    All three of these put the origin at the upstream edge of the working area,
    with `z` measured up from the belt surface. Every coordinate on the wire is
    MuJoCo world: the belt runs -1.50 m to +1.50 m, the arm base sits at
    (0, -0.70, 0.90), and the surface is at 0.90 m.

    The documents moved rather than the frame, because the golden vectors are
    frozen by rule and every measured coordinate is expressed against it. This
    fails if one of them is reverted.
    """
    for relative in BELT_FRAME_DOCUMENTS:
        text = (ROOT / relative).read_text()
        assert "upstream edge" not in text, relative
        assert "center of the belt" in text or "centre of the belt" in text, relative


def test_the_golden_vectors_were_not_touched_when_the_documents_were() -> None:
    """The golden vectors were not touched when the documents were.

    The other half, and the reason the documents moved instead of the frame. A
    vector that changes proves nothing, which its own README says, so correcting
    the prose had to leave every byte of these alone.
    """
    vectors = ROOT / "crates" / "clave-decision" / "contract" / "vectors" / "v1"
    assert sorted(p.name for p in vectors.glob("*.hex")) == [
        "nominal.hex",
        "rejected.hex",
        "unknown-version.hex",
    ]
    # The nominal vector's pose, which the belt frame correction would have
    # rewritten had the frame moved instead of the prose.
    readme = (vectors / "README.md").read_text()
    assert "0.412" in readme
    assert "0.031" in readme


def test_every_adapter_says_where_its_confidence_comes_from() -> None:
    """Every adapter says where its confidence comes from.

    An adapter reporting a bare constant tells a fusion rule how much to trust
    it without telling anybody why, and the weight it feeds is half of what
    the ban on source priority rests on.
    """
    from clave.tracker.adapters.code import codes_from_frame
    from clave.tracker.adapters.detection import _confidence_of

    def prose(obj: object) -> str:
        """A docstring with its line wrapping removed.

        Matching a phrase against a wrapped docstring finds nothing, which is a
        property of the formatter rather than of the documentation.
        """
        return " ".join((inspect.getdoc(obj) or "").split()).lower()

    derived = prose(_confidence_of)
    assert "pixel" in derived, "a derived confidence says what it derives from"
    assert "saturates" in derived, "and says how it behaves at the ends"

    constant = prose(codes_from_frame)
    assert "confidence" in constant
    # The code adapter does report a constant, so it owes a reason.
    assert "check digit" in constant, (
        "a constant confidence names why the constant is correct"
    )
