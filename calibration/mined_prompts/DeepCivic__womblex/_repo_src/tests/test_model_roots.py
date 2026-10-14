"""Plugin-supplied model roots resolve offline like bundled models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from womblex.utils import models


def _install(monkeypatch: pytest.MonkeyPatch, *values: object) -> None:
    eps = [SimpleNamespace(name=f"r{i}", load=lambda v=v: v) for i, v in enumerate(values)]
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: eps)
    models._plugin_roots.cache_clear()


@pytest.fixture(autouse=True)
def _reset():
    yield
    models._plugin_roots.cache_clear()


def test_plugin_root_resolves_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "my-model").mkdir()
    _install(monkeypatch, str(tmp_path))
    assert models.resolve_local_model_path("my-model", record=False) == tmp_path / "my-model"


def test_callable_and_iterable_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _install(monkeypatch, lambda: a, [b])
    roots = models.model_roots()
    assert a in roots and b in roots


def test_plugin_root_is_searched_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / "env"
    plug = tmp_path / "plug"
    for root in (env, plug):
        (root / "m").mkdir(parents=True)
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(env))
    _install(monkeypatch, plug)
    assert models.resolve_local_model_path("m", record=False) == env / "m"


def test_broken_entry_point_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("bad")

    _install(monkeypatch, boom, tmp_path)
    assert tmp_path in models.model_roots()


def test_installed_package_supplies_models_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real distribution on sys.path: registry entry points and a model root."""
    from womblex.ingest.paddle_ocr import get_ocr_reader, is_llm_engine
    from womblex.utils import model_registry as reg

    (tmp_path / "fakeplug_models" / "weights").mkdir(parents=True)
    (tmp_path / "fakeplug.py").write_text(
        "from pathlib import Path\n"
        "class Reader:\n"
        "    def __init__(self, lang='eng', **opts):\n"
        "        self.lang, self.opts = lang, opts\n"
        "def make_reader(lang='eng', **opts):\n"
        "    return Reader(lang, **opts)\n"
        "make_reader.womblex_traits = {'markdown': True}\n"
        "def models_dir():\n"
        "    return Path(__file__).parent / 'fakeplug_models'\n"
    )
    dist = tmp_path / "fakeplug-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: fakeplug\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(
        "[womblex.models.ocr]\nfake-ocr = fakeplug:make_reader\n"
        "[womblex.model_roots]\nfakeplug = fakeplug:models_dir\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for attr in ("_entries", "_aliases"):
        monkeypatch.setattr(
            reg, attr, {slot: dict(v) for slot, v in getattr(reg, attr).items()}
        )
    monkeypatch.setattr(reg, "_loaded", set())
    models._plugin_roots.cache_clear()

    assert "fake-ocr" in reg.known_names(reg.SLOT_OCR)
    entry = reg.resolve(reg.SLOT_OCR, "fake-ocr")
    assert entry.source == "fakeplug"
    assert is_llm_engine("fake-ocr")
    assert get_ocr_reader(engine="fake-ocr", k=1).opts == {"k": 1}
    assert models.resolve_local_model_path("weights", record=False) == (
        tmp_path / "fakeplug_models" / "weights"
    )
