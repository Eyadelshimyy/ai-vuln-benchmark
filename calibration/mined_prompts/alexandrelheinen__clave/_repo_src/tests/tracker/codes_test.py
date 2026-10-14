"""Reading a symbol, and resolving it to what the package is made of.



The decoder runs against a committed fixture, so a zero rendered yield reads
as a property of the optics rather than as breakage. The
fixture is a symbol on a pinned object texture: `docs/guidelines.md` says no
binary lands in git, and the manifest in `configs/assets/objects.yml` is a
reference rather than a binary.

Generating the symbol instead was tried and abandoned. OpenCV detects before it
decodes and will not detect a synthetic symbol at any module width, quiet zone
or canvas padding, so a generated fixture would have proved only that the
generator and the decoder disagree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clave.taxonomy import BY_ID
from clave.tracker.codes import (
    CatalogError,
    OpenCvDecoder,
    components_of,
    load_catalog,
)
from clave.tracker.evidence import Code, Evidence, Role
from clave.world.config import load, require

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs" / "perception" / "packaging.yml"
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"

# Decoded from this repository's own objects, never looked up.
POTTED_MEAT = "037600138727"
PUDDING = "043000204719"
SILICON = "790011130178"

FIXTURES = (
    (
        "third_party/ycb_sim/textures/008_pudding_box.png",
        PUDDING,
    ),
    (
        "third_party/scanned_objects/models/JarroSil_Activated_Silicon/texture.png",
        SILICON,
    ),
)
"""Symbols this project already ships, by submodule pin rather than as binaries."""


def read_fixture(relative: str) -> object:
    """Load one pinned texture, skipping when the submodule is not checked out."""
    import numpy as np
    from PIL import Image

    cv2 = pytest.importorskip("cv2")
    Image.MAX_IMAGE_PIXELS = None
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} needs its submodule checked out")
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


@pytest.mark.parametrize(("relative", "expected"), FIXTURES)
def test_the_decoder_reads_a_symbol_this_project_actually_ships(
    relative: str, expected: str
) -> None:
    """The decoder reads a symbol this project actually ships.

    The gate's proof that the decoder works. Without it, a measured rendered
    yield of a few percent could equally mean the optics are marginal or the
    decoder is broken, and those want different responses.
    """
    read = OpenCvDecoder().decode(read_fixture(relative))
    assert read is not None
    assert read.digits == expected


def test_a_symbol_too_coarse_to_resolve_decodes_to_nothing() -> None:
    """A symbol too coarse to resolve decodes to nothing.

    An EAN-13 module wants roughly two pixels. Below that the symbol is present
    and unreadable, which is a decline rather than a guess.
    """
    cv2 = pytest.importorskip("cv2")

    tiny = cv2.resize(read_fixture(FIXTURES[0][0]), None, fx=0.03, fy=0.03)
    assert OpenCvDecoder().decode(tiny) is None


def test_a_frame_with_no_symbol_decodes_to_nothing() -> None:
    """Most frames on a belt carry no readable symbol at all."""
    import numpy as np

    blank = np.full((120, 200, 3), 255, dtype=np.uint8)
    assert OpenCvDecoder().decode(blank) is None


def test_a_misread_symbol_is_refused_by_its_own_check_digit() -> None:
    """A misread symbol is refused by its own check digit.

    The refusal lives on the payload, so a `Code` that exists is one that
    checked out and no consumer has to ask. A misread that survived would
    resolve to some other product's bill of materials.
    """
    with pytest.raises(Exception, match="check digit"):
        Code(symbology="UPC_A", digits="037600138728")


def test_the_catalog_resolves_a_code_to_components_and_not_to_a_class() -> None:
    """The catalog resolves a code to components and not to a class.

    A bottle with a paper label returns HDPE and paper. That says one of
    them carries the symbol, not which, which is the distinction the contract
    insists on.
    """
    catalog = load_catalog(CATALOG)
    assert components_of(catalog, SILICON) == ("M-02", "M-09")


def test_an_unknown_code_resolves_to_nothing_rather_than_a_guess() -> None:
    """An unknown code resolves to nothing rather than a guess."""
    catalog = load_catalog(CATALOG)
    assert components_of(catalog, "012345678905") == ()


def test_every_catalogued_component_is_a_class_the_taxonomy_defines() -> None:
    """A component nothing can route to is not a component."""
    catalog = load_catalog(CATALOG)
    for entry in catalog.values():
        assert entry.components
        for component in entry.components:
            assert component in BY_ID, component


def test_every_catalogued_code_passes_its_own_check_digit() -> None:
    """Every catalogued code passes its own check digit.

    A catalogued code that could not have been read is a catalogued typo, and
    it would resolve a real misread to a plausible bill of materials.
    """
    from clave.tracker.evidence import check_digit_holds

    for digits in load_catalog(CATALOG):
        assert check_digit_holds(digits), digits


def test_every_catalogued_code_names_an_object_the_world_carries() -> None:
    """Every catalogued code names an object the world carries.

    The catalog records which package each symbol was decoded off, so an entry
    nobody can trace back to this repository's own object set is an entry
    somebody looked up.
    """
    named = {str(entry["name"]) for entry in require(load(WORLD), "objects")}
    for digits, entry in load_catalog(CATALOG).items():
        assert entry.object_name in named, f"{digits} names {entry.object_name}"


def test_a_catalogued_component_agrees_with_the_class_the_world_gives_it() -> None:
    """A catalogued component agrees with the class the world gives it.

    The bill of materials is inferred from what the package is, the same way
    the world infers `material_class`. They are allowed to differ, because a
    can's label is paper and its class is steel, but the world's class has to
    appear among the components or one of the two is wrong.
    """
    world = {
        str(entry["name"]): str(entry["material_class"])
        for entry in require(load(WORLD), "objects")
    }
    for digits, entry in load_catalog(CATALOG).items():
        assert world[entry.object_name] in entry.components, digits


def test_a_catalog_naming_a_class_outside_the_taxonomy_is_refused() -> None:
    """A catalog naming a class outside the taxonomy is refused."""
    import textwrap

    bad = ROOT / "configs" / "perception" / "_probe.yml"
    bad.write_text(
        textwrap.dedent(
            """
            codes:
              "037600138727":
                object: potted_meat_can
                decoded_from: texture
                components: [M-99]
            """
        )
    )
    try:
        with pytest.raises(CatalogError, match="M-99"):
            load_catalog(bad)
    finally:
        bad.unlink()


def test_the_adapter_turns_a_read_symbol_into_evidence() -> None:
    """The adapter turns a read symbol into evidence."""
    from clave.tracker.adapters.code import codes_from_frame

    readings = codes_from_frame(
        read_fixture(FIXTURES[0][0]),
        source_id="gate_code_center",
        observed_at_nanos=1_000,
        decoder=OpenCvDecoder(),
    )
    assert len(readings) == 1
    assert readings[0].role is Role.CODE
    assert readings[0].source_id == "gate_code_center"
    payload = readings[0].payload
    assert isinstance(payload, Code)
    assert payload.digits == PUDDING


def test_the_adapter_emits_nothing_when_nothing_decodes() -> None:
    """The adapter emits nothing when nothing decodes.

    A frame with no readable symbol produces no evidence, which is the ordinary
    case: the measured yield says most gate crossings decode nothing.
    """
    import numpy as np

    from clave.tracker.adapters.code import codes_from_frame

    blank = np.full((120, 200, 3), 255, dtype=np.uint8)
    assert (
        codes_from_frame(
            blank,
            source_id="gate_code_center",
            observed_at_nanos=1_000,
            decoder=OpenCvDecoder(),
        )
        == ()
    )


def test_a_code_reaching_a_track_moves_its_belief_toward_the_components() -> None:
    """A code reaching a track moves its belief toward the components."""
    from clave.tracker.codes import resolver_for
    from clave.tracker.fusion import FusionSettings, Posterior, fold

    settings = FusionSettings.load(
        ROOT / "configs" / "perception" / "fusion.yml", WORLD
    )
    reading = _wrap(Code(symbology="UPC_A", digits=SILICON))
    after = fold(
        Posterior.uniform(),
        reading,
        1_000,
        settings,
        resolve=resolver_for(load_catalog(CATALOG)),
    )
    assert after.recognized
    assert after.posterior.most_likely[0] in ("M-02", "M-09")


def _wrap(payload: Code) -> Evidence:
    """An envelope around one code, from a camera that reads codes."""
    return Evidence(
        source_id="gate_code_center",
        role=Role.CODE,
        observed_at_nanos=1_000,
        confidence=0.9,
        payload=payload,
    )
