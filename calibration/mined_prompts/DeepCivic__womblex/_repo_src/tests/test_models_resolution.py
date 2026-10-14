"""Local-model resolution: per-artefact search across every models root.

The regression this pins is the one the container image hit and the suite
missed: ``WOMBLEX_MODELS_DIR`` points at a root holding only the large
artefacts, so a single-root resolver stopped finding the wheel-bundled
``en_AU`` dictionary and spellfix died inside spylls on the relative path
``en_AU/index.aff``. The override must *supplement* the bundled root, not
shadow it — and an unresolved dictionary must fail with an actionable message
rather than a FileNotFoundError several frames away.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import womblex
from womblex.utils.models import (
    digest_model_path,
    find_models_dir,
    loaded_models,
    model_roots,
    record_loaded_path,
    reset_loaded_models,
    resolve_local_model_path,
)

#: The wheel-bundled root, taken from the installed package (not the repo
#: layout) so the test pins what a `pip install womblex` actually resolves.
BUNDLED = Path(womblex.__file__).resolve().parent / "_models"


@pytest.fixture
def only_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A WOMBLEX_MODELS_DIR holding the large artefacts and nothing else."""
    root = tmp_path / "app-models"
    (root / "all-MiniLM-L6-v2").mkdir(parents=True)
    (root / "large-artefact.bin").write_bytes(b"weights")
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(root))
    return root


def test_env_root_does_not_shadow_bundled_dictionary(only_env_root: Path):
    """en_AU still resolves when the override root does not carry it."""
    resolved = resolve_local_model_path("en_AU")
    assert isinstance(resolved, Path), "en_AU fell back to the bare string"
    assert resolved == BUNDLED / "en_AU"
    assert (resolved / "index.aff").is_file()


def test_env_root_still_wins_for_what_it_holds(only_env_root: Path):
    """The override keeps priority for the artefacts it does carry."""
    assert resolve_local_model_path("all-MiniLM-L6-v2") == only_env_root / "all-MiniLM-L6-v2"
    assert resolve_local_model_path("large-artefact.bin") == only_env_root / "large-artefact.bin"


def test_roots_are_ordered_and_deduplicated(only_env_root: Path):
    roots = model_roots()
    assert roots[0] == only_env_root
    assert BUNDLED in roots
    assert len(roots) == len(set(roots))
    assert find_models_dir() == only_env_root


def test_bundled_dictionary_loads_under_env_override(only_env_root: Path):
    """The end-to-end symptom: spellfix's dictionary loader must not raise."""
    pytest.importorskip("spylls")
    from womblex.process import spellfix

    assert spellfix._dictionary("en_AU").lookup("child")


def test_missing_dictionary_raises_actionable_error(only_env_root: Path):
    """A bare-string return must not reach spylls as a relative path."""
    pytest.importorskip("spylls")
    from womblex.process import spellfix

    with pytest.raises(FileNotFoundError, match="no models root holds"):
        spellfix._dictionary("en_ZZ")


def test_hub_snapshot_layout_still_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """refs/main → snapshots/<hash>/ is unchanged by the multi-root search."""
    root = tmp_path / "models"
    model = root / "some-model"
    (model / "refs").mkdir(parents=True)
    (model / "refs" / "main").write_text("deadbeef\n")
    snapshot = model / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(root))

    assert resolve_local_model_path("some-model") == snapshot


def test_dangling_refs_main_falls_back_to_the_flat_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A refs/main naming a snapshot that is not there returns the model dir."""
    root = tmp_path / "models"
    model = root / "some-model"
    (model / "refs").mkdir(parents=True)
    (model / "refs" / "main").write_text("missing")
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(root))

    assert resolve_local_model_path("some-model") == model


def test_unknown_artefact_echoes_the_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(tmp_path))
    assert resolve_local_model_path("org/not-vendored") == "org/not-vendored"


def test_nonexistent_env_root_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A typo'd override must not remove the bundled root from the search."""
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(tmp_path / "nope"))
    assert resolve_local_model_path("en_AU") == BUNDLED / "en_AU"


class TestLoadRecord:
    """What a run loaded, recorded at the one choke point every load goes through.

    Which local model produced an output is not recoverable after the fact — a
    model directory is swapped in place far more often than it is renamed — so
    resolution records, and the run stamp carries the record into each footer.
    """

    @pytest.fixture(autouse=True)
    def _clean_record(self):
        reset_loaded_models()
        yield
        reset_loaded_models()

    def test_a_resolution_is_recorded_with_a_digest(self):
        resolve_local_model_path("en_AU")
        assert [m.name for m in loaded_models()] == ["en_AU"]
        assert loaded_models()[0].digest.startswith("sha256:")

    def test_a_probe_is_not_a_load(self):
        """`record=False` asks whether an artefact is present without claiming
        the run used it — the record says what was loaded."""
        assert isinstance(resolve_local_model_path("en_AU", record=False), Path)
        assert loaded_models() == ()

    def test_a_name_that_does_not_resolve_records_nothing(self):
        assert resolve_local_model_path("no-such-model") == "no-such-model"
        assert loaded_models() == ()

    def test_a_library_bundled_artefact_can_be_recorded_too(self, tmp_path: Path):
        """RapidOCR loads its fallback models from inside its own wheel, so they
        never reach the resolver — the loader records them explicitly instead,
        or the record shows no OCR model for a run that OCR'd."""
        bundled = tmp_path / "wheel" / "models"
        bundled.mkdir(parents=True)
        (bundled / "det.onnx").write_bytes(b"weights")
        record_loaded_path("rapidocr-bundled-v4", bundled)
        assert [m.name for m in loaded_models()] == ["rapidocr-bundled-v4"]
        assert loaded_models()[0].digest == digest_model_path(bundled)

    def test_an_absent_path_is_not_recorded_undigestable(self, tmp_path: Path):
        record_loaded_path("gone", tmp_path / "nowhere")
        assert loaded_models() == ()

    def test_the_record_is_name_sorted_not_load_ordered(self):
        """A run's record must not depend on which stage happened to run first."""
        resolve_local_model_path("kanon-2-tokenizer")
        resolve_local_model_path("en_AU")
        assert [m.name for m in loaded_models()] == ["en_AU", "kanon-2-tokenizer"]


class TestDigestRecomputes:
    """The acceptance criterion: the digest recomputes from the model files alone."""

    @pytest.fixture
    def artefact(self, tmp_path: Path) -> Path:
        root = tmp_path / "a" / "model"
        (root / "nested").mkdir(parents=True)
        (root / "weights.bin").write_bytes(b"\x00\x01\x02")
        (root / "nested" / "config.json").write_text('{"k": 1}')
        return root

    def test_a_copy_elsewhere_digests_the_same(self, artefact: Path, tmp_path: Path):
        """No path, machine or run is folded in — only the bytes."""
        copy = tmp_path / "b" / "model"
        shutil.copytree(artefact, copy)
        assert digest_model_path(copy) == digest_model_path(artefact)

    def test_changed_content_changes_the_digest(self, artefact: Path, tmp_path: Path):
        changed = tmp_path / "c" / "model"
        shutil.copytree(artefact, changed)
        (changed / "weights.bin").write_bytes(b"\x00\x01\x03")
        assert digest_model_path(changed) != digest_model_path(artefact)

    def test_a_rename_inside_the_tree_changes_the_digest(self, artefact: Path, tmp_path: Path):
        """Relative paths are digested as well as bytes, so a swap of two
        same-sized files is a different artefact rather than the same one."""
        renamed = tmp_path / "d" / "model"
        shutil.copytree(artefact, renamed)
        (renamed / "weights.bin").rename(renamed / "weights.safetensors")
        assert digest_model_path(renamed) != digest_model_path(artefact)

    def test_a_single_file_artefact_digests_its_bytes(self, tmp_path: Path):
        one = tmp_path / "model.bin"
        one.write_bytes(b"weights")
        two = tmp_path / "elsewhere" / "model.bin"
        two.parent.mkdir()
        two.write_bytes(b"weights")
        assert digest_model_path(one) == digest_model_path(two)
