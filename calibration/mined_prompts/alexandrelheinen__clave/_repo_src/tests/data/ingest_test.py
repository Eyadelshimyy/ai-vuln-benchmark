"""Tests for reading a fetched corpus archive into labeled examples."""

import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.corpus.artifacts import digest_of
from clave.corpus.manifest import Manifest
from clave.data.examples import Origin
from clave.data.ingest import (
    CORPUS_LAYOUTS,
    CorpusArchive,
    CorpusError,
    LocalizedArchive,
    compose_corpus,
    compose_localized,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRASHNET = ROOT / "datasets" / "corpora" / "trashnet" / "dataset-resized.zip"

FIXTURE_MEMBERS = {
    "dataset-resized/": b"",
    "dataset-resized/.DS_Store": b"editor metadata",
    "dataset-resized/glass/": b"",
    "dataset-resized/glass/.DS_Store": b"editor metadata inside a label",
    "dataset-resized/glass/notes.txt": b"a label directory may hold prose",
    "dataset-resized/glass/glass1.jpg": b"glass pixels",
    "dataset-resized/glass/glass2.jpg": b"more glass pixels",
    "dataset-resized/plastic/plastic1.jpg": b"plastic pixels",
    "dataset-resized/unicorn/unicorn1.jpg": b"unmapped pixels",
    "dataset-resized/stray.jpg": b"outside any label directory",
    "__MACOSX/dataset-resized/glass/._glass1.jpg": b"resource fork",
}


def trashnet_archive() -> Path | None:
    """Return the fetched TrashNet archive, or None when it is absent.

    An operator fetches the corpus; the gate does not. Everything below that
    needs the real bytes asks here and skips when the answer is None.
    """

    override = os.environ.get("CLAVE_TRASHNET_ARCHIVE")
    path = Path(override) if override else DEFAULT_TRASHNET
    return path if path.is_file() else None


def build_archive(path: Path, members: dict[str, bytes]) -> Path:
    """Write a zip holding the given members, and return its path."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


@pytest.fixture
def fixture_archive(tmp_path: Path) -> Path:
    """A small archive laid out the way TrashNet lays out its own."""
    return build_archive(tmp_path / "trashnet.zip", FIXTURE_MEMBERS)


def test_a_fetched_archive_reads_into_labeled_examples(fixture_archive: Path) -> None:
    """Imagery is read from the archive without extracting it."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        members = [example.member for example in archive.examples]
    assert members == [
        "dataset-resized/glass/glass1.jpg",
        "dataset-resized/glass/glass2.jpg",
        "dataset-resized/plastic/plastic1.jpg",
        "dataset-resized/unicorn/unicorn1.jpg",
    ]


def test_every_fetched_example_carries_the_real_origin(fixture_archive: Path) -> None:
    """Real examples stay separable."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        origins = {example.origin for example in archive.examples}
    assert origins == {Origin.REAL}


def test_a_label_spanning_several_classes_is_read_as_ambiguous(
    fixture_archive: Path,
) -> None:
    """TrashNet's plastic covers four classes."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        plastic = next(
            example for example in archive.examples if example.source_label == "plastic"
        )
    assert plastic.label.ambiguous
    assert plastic.label.classes == ("M-01", "M-02", "M-03", "M-04")


def test_a_label_outside_the_mapping_is_read_as_unmapped(
    fixture_archive: Path,
) -> None:
    """An unrecognized directory does not lose its images."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        unicorn = next(
            example for example in archive.examples if example.source_label == "unicorn"
        )
    assert not unicorn.label.mapped
    assert unicorn.label.classes == ()


def test_editor_metadata_is_not_read_as_imagery(fixture_archive: Path) -> None:
    """AppleDouble forks and .DS_Store files are not images.

    Three separate rules exclude them, so the fixture carries one member for
    each: outside the declared root, directly under the root, and hidden inside
    a label directory.
    """
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        members = [example.member for example in archive.examples]
    assert not [member for member in members if "__MACOSX" in member]
    assert not [member for member in members if "/." in member]
    assert "dataset-resized/glass/.DS_Store" not in members


def test_a_file_outside_a_label_directory_is_not_read(
    fixture_archive: Path,
) -> None:
    """A member at the archive root carries no label, so it is not an example.

    Nor does a directory entry or a file whose suffix the layout does not
    declare, both of which sit in the right place and hold no imagery.
    """
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        members = [example.member for example in archive.examples]
    assert "dataset-resized/stray.jpg" not in members
    assert "dataset-resized/glass/" not in members
    assert "dataset-resized/glass/notes.txt" not in members


def test_a_corpus_with_no_declared_layout_is_refused(fixture_archive: Path) -> None:
    """Guessing a layout would invent a label vocabulary."""
    with pytest.raises(CorpusError, match="taco"):
        CorpusArchive(fixture_archive, "taco")


def test_an_archive_holding_no_imagery_is_refused(tmp_path: Path) -> None:
    """An empty read is a failure, not an empty corpus."""
    empty = build_archive(tmp_path / "empty.zip", {"readme.txt": b"nothing here"})
    with pytest.raises(CorpusError, match="dataset-resized"):
        CorpusArchive(empty, "trashnet")


def test_a_file_that_is_not_an_archive_is_refused(tmp_path: Path) -> None:
    """A truncated download must fail loudly rather than read as empty."""
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip at all")
    with pytest.raises(CorpusError, match="readable archive"):
        CorpusArchive(broken, "trashnet")


def test_a_frame_decodes_through_the_caller_supplied_decoder(
    fixture_archive: Path,
) -> None:
    """Ingestion needs no image library of its own."""
    seen: list[bytes] = []

    def decode(payload: bytes) -> NDArray[np.uint8]:
        seen.append(payload)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    with CorpusArchive(fixture_archive, "trashnet") as archive:
        first = archive.examples[0]
        assert archive.read_bytes(first) == b"glass pixels"
        frame = archive.read_frame(first, decode)
    assert seen == [b"glass pixels"]
    assert frame.shape == (2, 2, 3)


def test_composition_counts_every_class_a_label_may_be(
    fixture_archive: Path,
) -> None:
    """An ambiguous label counts toward each class it spans."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        measured = compose_corpus("trashnet", archive.examples)
    assert measured.example_count == 4
    assert measured.source_counts == {"glass": 2, "plastic": 1, "unicorn": 1}
    assert measured.class_counts["M-07"] == 2
    assert measured.class_counts["M-01"] == 1
    assert measured.class_counts["M-04"] == 1
    assert measured.ambiguous_count == 1
    assert measured.unmapped_count == 1


def test_composition_separates_resolved_classes_from_ambiguous_ones(
    fixture_archive: Path,
) -> None:
    """Only a label pinning one class resolves that class."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        measured = compose_corpus("trashnet", archive.examples)
    assert measured.resolved_class_counts == {"M-07": 2}
    assert measured.unresolved_classes == ("M-01", "M-02", "M-03", "M-04")


def test_composition_names_the_classes_a_corpus_cannot_supply(
    fixture_archive: Path,
) -> None:
    """An absent class is an absence in the data."""
    with CorpusArchive(fixture_archive, "trashnet") as archive:
        measured = compose_corpus("trashnet", archive.examples)
    assert "M-10" in measured.absent_classes
    assert "M-07" not in measured.absent_classes


def test_trashnet_declares_an_archive_layout() -> None:
    """The fetched corpus is the one whose layout the module describes."""
    assert CORPUS_LAYOUTS["trashnet"].root == "dataset-resized"


@pytest.mark.skipif(
    trashnet_archive() is None, reason="TrashNet has not been fetched locally"
)
def test_the_fetched_trashnet_archive_matches_its_manifest_digest() -> None:
    """The committed digest describes the bytes this machine holds."""
    path = trashnet_archive()
    assert path is not None
    recorded = Manifest.load(ROOT / "corpora" / "manifest.toml")["trashnet"]
    assert digest_of(path) == recorded.sha256


@pytest.mark.skipif(
    trashnet_archive() is None, reason="TrashNet has not been fetched locally"
)
def test_the_fetched_trashnet_archive_holds_its_recorded_composition() -> None:
    """The ingestion counts come from these bytes."""
    path = trashnet_archive()
    assert path is not None
    with CorpusArchive(path, "trashnet") as archive:
        measured = compose_corpus("trashnet", archive.examples)
    assert measured.example_count == 2527
    assert measured.source_counts == {
        "cardboard": 403,
        "glass": 501,
        "metal": 410,
        "paper": 594,
        "plastic": 482,
        "trash": 137,
    }
    assert measured.unmapped_count == 0
    assert measured.absent_classes == ("M-10",)
    assert measured.unresolved_classes == (
        "M-01",
        "M-02",
        "M-03",
        "M-04",
        "M-05",
        "M-06",
    )


ZEROWASTE_ROOT = "splits_final_deblurred"

DEFAULT_ZEROWASTE = (
    ROOT / "datasets" / "corpora" / "zerowaste" / "zerowaste-f-final.zip"
)


def zerowaste_archive() -> Path | None:
    """Return the fetched ZeroWaste archive, or None when it is absent."""
    override = os.environ.get("CLAVE_ZEROWASTE_ARCHIVE")
    path = Path(override) if override else DEFAULT_ZEROWASTE
    return path if path.is_file() else None


def coco_labels(
    images: list[dict[str, object]], annotations: list[dict[str, object]]
) -> bytes:
    """Render one COCO annotation file, with ZeroWaste's four categories."""
    return json.dumps(
        {
            "categories": [
                {"id": 1, "name": "rigid_plastic"},
                {"id": 2, "name": "cardboard"},
                {"id": 3, "name": "metal"},
                {"id": 4, "name": "soft_plastic"},
            ],
            "images": images,
            "annotations": annotations,
        }
    ).encode("utf-8")


def localized_fixture(path: Path, **overrides: bytes) -> Path:
    """Build a small archive shaped like ZeroWaste, with all three splits."""
    members = {
        f"{ZEROWASTE_ROOT}/train/data/a.PNG": b"first frame",
        f"{ZEROWASTE_ROOT}/train/data/b.PNG": b"second frame",
        f"{ZEROWASTE_ROOT}/train/sem_seg/a.PNG": b"a mask, not imagery",
        f"{ZEROWASTE_ROOT}/train/labels.json": coco_labels(
            images=[
                {"id": 0, "file_name": "a.PNG", "width": 1920, "height": 1080},
                {"id": 1, "file_name": "b.PNG", "width": 1920, "height": 1080},
            ],
            annotations=[
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 2,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "area": 1200.0,
                },
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 4,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "area": 25.0,
                },
            ],
        ),
        f"{ZEROWASTE_ROOT}/val/data/c.PNG": b"third frame",
        f"{ZEROWASTE_ROOT}/val/labels.json": coco_labels(
            images=[{"id": 0, "file_name": "c.PNG", "width": 1920, "height": 1080}],
            annotations=[
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 3,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "area": 12.0,
                }
            ],
        ),
        f"{ZEROWASTE_ROOT}/test/data/d.PNG": b"fourth frame",
        f"{ZEROWASTE_ROOT}/test/labels.json": coco_labels(
            images=[{"id": 0, "file_name": "d.PNG", "width": 1920, "height": 1080}],
            annotations=[
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [5.0, 5.0, 5.0, 5.0],
                    "area": 25.0,
                }
            ],
        ),
    }
    members.update(overrides)
    return build_archive(path, members)


def test_a_localized_corpus_reads_regions_with_boxes_and_mapped_labels(
    tmp_path: Path,
) -> None:
    """The localization is the reason this corpus outranks the rest."""
    archive_path = localized_fixture(tmp_path / "zerowaste.zip")
    with LocalizedArchive(archive_path, "zerowaste") as archive:
        examples = archive.examples
    assert [example.member for example in examples] == [
        f"{ZEROWASTE_ROOT}/train/data/a.PNG",
        f"{ZEROWASTE_ROOT}/train/data/b.PNG",
        f"{ZEROWASTE_ROOT}/val/data/c.PNG",
        f"{ZEROWASTE_ROOT}/test/data/d.PNG",
    ]
    first = examples[0]
    assert first.split == "train"
    assert first.origin is Origin.REAL
    assert len(first.regions) == 2
    cardboard = first.regions[0]
    assert cardboard.label.source_label == "cardboard"
    assert cardboard.label.classes == ("M-08", "M-09", "M-10")
    assert cardboard.label.ambiguous
    # COCO writes a box as x, y, width, height; CLAVE carries corners.
    assert cardboard.bbox == (10.0, 20.0, 40.0, 60.0)
    assert first.regions[1].label.classes == ("M-04",)
    assert not first.regions[1].label.ambiguous


def test_an_image_with_no_annotation_is_kept_and_reported(tmp_path: Path) -> None:
    """A frame with no foreground is a real frame, not a gap."""
    archive_path = localized_fixture(tmp_path / "zerowaste.zip")
    with LocalizedArchive(archive_path, "zerowaste") as archive:
        composition = compose_localized("zerowaste", archive.examples)
        empty = [example for example in archive.examples if not example.regions]
    assert [example.member for example in empty] == [
        f"{ZEROWASTE_ROOT}/train/data/b.PNG"
    ]
    assert composition.example_count == 4
    assert composition.unannotated_count == 1


def test_an_annotation_naming_an_undeclared_category_is_refused(
    tmp_path: Path,
) -> None:
    """A category the file never declares is corruption, not a label."""
    archive_path = localized_fixture(
        tmp_path / "zerowaste.zip",
        **{
            f"{ZEROWASTE_ROOT}/train/labels.json": coco_labels(
                images=[
                    {"id": 0, "file_name": "a.PNG", "width": 1920, "height": 1080},
                    {"id": 1, "file_name": "b.PNG", "width": 1920, "height": 1080},
                ],
                annotations=[
                    {
                        "id": 0,
                        "image_id": 0,
                        "category_id": 99,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 4.0,
                    }
                ],
            )
        },
    )
    with pytest.raises(CorpusError, match="category 99"):
        LocalizedArchive(archive_path, "zerowaste")


def test_an_annotated_image_missing_from_the_archive_is_refused(
    tmp_path: Path,
) -> None:
    """Reading fewer images than the annotations describe is a fault."""
    archive_path = localized_fixture(
        tmp_path / "zerowaste.zip",
        **{
            f"{ZEROWASTE_ROOT}/train/labels.json": coco_labels(
                images=[
                    {"id": 0, "file_name": "a.PNG", "width": 1920, "height": 1080},
                    {"id": 7, "file_name": "ghost.PNG", "width": 1920, "height": 1080},
                ],
                annotations=[],
            )
        },
    )
    with pytest.raises(CorpusError, match="ghost.PNG"):
        LocalizedArchive(archive_path, "zerowaste")


def test_a_localized_composition_counts_regions_rather_than_images(
    tmp_path: Path,
) -> None:
    """One image holding six objects is six pieces of evidence."""
    archive_path = localized_fixture(tmp_path / "zerowaste.zip")
    with LocalizedArchive(archive_path, "zerowaste") as archive:
        composition = compose_localized("zerowaste", archive.examples)
    assert composition.example_count == 4
    assert composition.region_count == 4
    assert composition.source_counts == {
        "cardboard": 1,
        "metal": 1,
        "rigid_plastic": 1,
        "soft_plastic": 1,
    }
    assert composition.resolved_class_counts == {"M-04": 1}
    # One rigid_plastic region spans M-01 through M-04, so M-04 is counted
    # twice: once resolved from soft_plastic and once as a class the rigid
    # label may be.
    assert composition.class_counts == {
        "M-01": 1,
        "M-02": 1,
        "M-03": 1,
        "M-04": 2,
        "M-05": 1,
        "M-06": 1,
        "M-08": 1,
        "M-09": 1,
        "M-10": 1,
    }
    assert composition.ambiguous_count == 3
    assert "M-07" in composition.absent_classes
    assert "M-08" in composition.unresolved_classes


def test_a_corpus_with_no_declared_coco_layout_is_refused(tmp_path: Path) -> None:
    """The reader infers no structure it was not told about."""
    archive_path = localized_fixture(tmp_path / "zerowaste.zip")
    with pytest.raises(CorpusError, match="declares no annotation layout"):
        LocalizedArchive(archive_path, "trashnet")


def test_a_localized_frame_decodes_through_the_caller_supplied_decoder(
    tmp_path: Path,
) -> None:
    """Ingestion still needs no image library, localized or not."""

    def decode(payload: bytes) -> NDArray[np.uint8]:
        assert payload == b"first frame"
        return np.zeros((2, 2, 3), dtype=np.uint8)

    archive_path = localized_fixture(tmp_path / "zerowaste.zip")
    with LocalizedArchive(archive_path, "zerowaste") as archive:
        example = archive.examples[0]
        assert archive.read_bytes(example) == b"first frame"
        assert archive.read_frame(example, decode).shape == (2, 2, 3)


@pytest.mark.skipif(
    zerowaste_archive() is None, reason="ZeroWaste has not been fetched locally"
)
def test_the_fetched_zerowaste_archive_matches_its_manifest_digest() -> None:
    """The committed digest describes the bytes this machine holds."""
    path = zerowaste_archive()
    assert path is not None
    recorded = Manifest.load(ROOT / "corpora" / "manifest.toml")["zerowaste"]
    assert digest_of(path) == recorded.sha256


@pytest.mark.skipif(
    zerowaste_archive() is None, reason="ZeroWaste has not been fetched locally"
)
def test_the_fetched_zerowaste_archive_holds_its_recorded_composition() -> None:
    """The ingestion counts come from these bytes."""
    path = zerowaste_archive()
    assert path is not None
    with LocalizedArchive(path, "zerowaste") as archive:
        measured = compose_localized("zerowaste", archive.examples)
        splits = Counter(example.split for example in archive.examples)
    assert measured.example_count == 4503
    assert measured.region_count == 26766
    assert dict(splits) == {"train": 3002, "val": 572, "test": 929}
    assert measured.source_counts == {
        "cardboard": 17751,
        "metal": 382,
        "rigid_plastic": 1769,
        "soft_plastic": 6864,
    }
    assert measured.unannotated_count == 86
    assert measured.unmapped_count == 0
    assert measured.absent_classes == ("M-07", "M-11")
    assert measured.resolved_class_counts == {"M-04": 6864}


@pytest.mark.skipif(
    zerowaste_archive() is None, reason="ZeroWaste has not been fetched locally"
)
def test_a_zerowaste_file_name_does_not_identify_an_image_across_splits() -> None:
    """Three names appear in two splits each, naming different frames."""
    path = zerowaste_archive()
    assert path is not None
    with LocalizedArchive(path, "zerowaste") as archive:
        by_name: dict[str, set[str]] = {}
        for example in archive.examples:
            by_name.setdefault(example.file_name, set()).add(example.split)
    collisions = {name for name, splits in by_name.items() if len(splits) > 1}
    assert collisions == {
        "01_frame_049000.PNG",
        "03_frame_049000.PNG",
        "09_frame_003000.PNG",
    }
