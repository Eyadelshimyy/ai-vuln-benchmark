"""Model registry: name resolution, plugin entry points, error contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from womblex.ingest.paddle_ocr import get_ocr_reader, is_llm_engine
from womblex.utils import model_registry as reg


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    """Isolate registrations from the process-wide registry."""
    for attr in ("_entries", "_aliases"):
        monkeypatch.setattr(
            reg, attr, {slot: dict(v) for slot, v in getattr(reg, attr).items()}
        )
    monkeypatch.setattr(reg, "_loaded", set(reg._loaded))
    monkeypatch.setattr(reg, "_used", dict(reg._used))


def test_builtin_names_and_aliases_keep_resolving() -> None:
    for alias, canonical in {
        "paddle": "paddleocr", "rapidocr": "paddleocr", "PaddleOCR": "paddleocr",
        "mistral": "mistral-ocr", "pixtral": "mistral-ocr", "bedrock": "mistral-ocr",
        "ollama-ocr": "ollama",
    }.items():
        assert reg.resolve(reg.SLOT_OCR, alias).name == canonical


def test_markdown_trait_marks_llm_engines() -> None:
    assert is_llm_engine("pixtral") and is_llm_engine("ollama")
    assert not is_llm_engine("rapidocr")


def test_unknown_name_lists_known_names() -> None:
    with pytest.raises(ValueError, match="paddleocr") as exc:
        get_ocr_reader(engine="nope")
    assert "mistral-ocr" in str(exc.value)


@pytest.mark.parametrize("spec", ["pkg.mod:make", "pkg.mod", "a/b"])
def test_import_paths_are_refused(spec: str) -> None:
    with pytest.raises(ValueError, match="import paths are not accepted"):
        reg.resolve(reg.SLOT_OCR, spec)


def test_plugin_entry_point_is_selectable_by_name(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(lang: str = "eng", **opts):
        return ("reader", lang, opts)

    factory.womblex_traits = {"markdown": True}  # type: ignore[attr-defined]
    ep = SimpleNamespace(
        name="my-ocr", load=lambda: factory, dist=SimpleNamespace(name="my-pkg")
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [])

    assert "my-ocr" in reg.known_names(reg.SLOT_OCR)
    assert get_ocr_reader(engine="my-ocr", lang="fra", beam=3) == ("reader", "fra", {"beam": 3})
    assert is_llm_engine("my-ocr")
    assert reg.resolve(reg.SLOT_OCR, "my-ocr").source == "my-pkg"


@pytest.mark.parametrize("name", ["pixtral", "paddleocr"])
def test_plugin_cannot_shadow_a_builtin(
    clean_registry, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    ep = SimpleNamespace(name=name, load=lambda: object, dist=None)
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [])
    assert reg.resolve(reg.SLOT_OCR, name).source == "builtin"


# --- layout slot -----------------------------------------------------------

from womblex.ingest.interfaces.protocols import (
    LAYOUT_BLOCK_TYPES,
    LayoutRegionResult,
    check_layout_regions,
)
from womblex.ingest.layout_onnx import LABEL_MAP
from womblex.ingest.paddle_ocr import get_layout_analyzer


def _region(block_type: str = "paragraph", y0: float = 0.0, conf: float = 0.9):
    return LayoutRegionResult((0, y0, 10, y0 + 5), "x", block_type, conf)


def test_layout_default_name_and_alias_resolve() -> None:
    assert reg.resolve(reg.SLOT_LAYOUT, "PP-DocLayout").name == "pp-doclayout-m"


def test_layout_unknown_name_lists_known_names() -> None:
    with pytest.raises(ValueError, match="pp-doclayout-m"):
        get_layout_analyzer("nope")


def test_builtin_label_map_conforms_to_vocabulary() -> None:
    assert set(LABEL_MAP.values()) <= LAYOUT_BLOCK_TYPES


def test_conformance_accepts_a_good_result_and_empty() -> None:
    check_layout_regions([])
    check_layout_regions([_region("table", 0), _region("figure", 20)])


@pytest.mark.parametrize(
    "regions",
    [
        [_region("textbox")],
        [LayoutRegionResult((5, 5, 1, 1), "x", "table", 0.5)],
        [_region(conf=1.5)],
        [_region(y0=50), _region(y0=10)],
    ],
)
def test_conformance_rejects_bad_results(regions) -> None:
    with pytest.raises(ValueError):
        check_layout_regions(regions)


def test_plugin_layout_model_is_selectable_and_receives_options(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fake:
        def __init__(self, **opts):
            self.opts = opts

    ep = SimpleNamespace(name="my-layout", load=lambda: Fake, dist=SimpleNamespace(name="p"))
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".layout") else [])
    a = get_layout_analyzer("my-layout", size=3)
    assert a.opts == {"size": 3}  # type: ignore[attr-defined]
    assert get_layout_analyzer("my-layout", size=3) is a


# --- pii-context slot ------------------------------------------------------

import numpy as np

from womblex.pii.cleaner import PIICleaner


def test_pii_context_default_and_alias_resolve() -> None:
    entry = reg.resolve(reg.SLOT_PII_CONTEXT, "Sentence-Transformers/all-MiniLM-L6-v2")
    assert entry.name == "all-minilm-l6-v2"


def test_pii_context_unknown_name_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="all-minilm-l6-v2"):
        PIICleaner(model="nope")


def test_plugin_context_model_scores_candidates(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    class Encoder:
        def __init__(self, **opts):
            seen.update(opts)

        def encode(self, texts):
            # every text maps to the same direction: cosine similarity 1.0
            return np.ones((len(texts), 4))

    ep = SimpleNamespace(name="my-ctx", load=lambda: Encoder, dist=SimpleNamespace(name="p"))
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points", lambda group: [ep] if group.endswith(".pii-context") else []
    )
    cleaner = PIICleaner(model="my-ctx", model_options={"dim": 4})
    import re

    text = "Signed by Janine Fairburn today."
    match = re.search("Janine Fairburn", text)
    assert cleaner._score_context_batch(text, [match]) == pytest.approx([1.0])  # type: ignore[list-item]
    assert seen == {"dim": 4}


# --- redaction layout filter -----------------------------------------------


from womblex.redact.stage import _layout_exclude_rects


def _page_layout(status: str, *regions: tuple[str, tuple[float, float, float, float]]):
    from womblex.ingest.elements import BBox
    from womblex.ingest.layout_step import LayoutRegion, PageLayout

    return PageLayout(
        0, status, [LayoutRegion(BBox(*box), "x", block_type, 0.9) for block_type, box in regions],
        error="boom" if status == "error" else "",
    )


def test_redaction_filter_excludes_only_figures_and_tables_in_render_pixels() -> None:
    layout = _page_layout(
        "ok",
        ("table", (0.1, 0.2, 0.5, 0.25)),
        ("paragraph", (0.0, 0.0, 1.0, 0.1)),
        ("figure", (0.5, 0.5, 0.5, 0.5)),
    )
    rects = _layout_exclude_rects(np.zeros((200, 100, 3), dtype=np.uint8), layout)
    assert rects == [(10, 40, 60, 90), (50, 100, 100, 200)]


def test_redaction_filter_treats_an_empty_page_as_nothing_to_exclude() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert _layout_exclude_rects(img, _page_layout("empty")) == []


@pytest.mark.parametrize("layout", [None, _page_layout("error")])
def test_redaction_filter_has_nothing_without_usable_layout(layout) -> None:
    assert _layout_exclude_rects(np.zeros((10, 10, 3), dtype=np.uint8), layout) is None


# --- tokenizer and spellfix-dictionary slots -------------------------------

from womblex.process.chunker import resolve_tokenizer
from womblex.process.spellfix import repair_text


def test_tokenizer_builtins_resolve() -> None:
    assert resolve_tokenizer("isaacus/kanon-2-tokenizer") == "isaacus/kanon-2-tokenizer"
    assert resolve_tokenizer("kanon-2-tokenizer") == "isaacus/kanon-2-tokenizer"
    assert resolve_tokenizer("huggingface", {"name": "org/tok"}) == "org/tok"


def test_tokenizer_unknown_name_lists_known_names() -> None:
    with pytest.raises(ValueError, match="huggingface") as exc:
        resolve_tokenizer("gpt2")
    assert "kanon-2-tokenizer" in str(exc.value)


def test_plugin_tokenizer_may_be_a_token_counter(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make(**opts):
        return lambda text: len(text.split())

    ep = SimpleNamespace(name="words", load=lambda: make, dist=SimpleNamespace(name="p"))
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points", lambda group: [ep] if group.endswith(".tokenizer") else []
    )
    assert resolve_tokenizer("words")("a b c") == 3


def test_spellfix_builtin_dictionary_names_resolve() -> None:
    for name in ("en_AU", "en-au", "hunspell"):
        reg.resolve(reg.SLOT_SPELLFIX_DICTIONARY, name)


def test_spellfix_unknown_dictionary_lists_known_names() -> None:
    with pytest.raises(ValueError, match="en_au"):
        repair_text("The chi1d went home.", dict_name="en_ZZ")


def test_plugin_dictionary_receives_options(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Words:
        def __init__(self, extra: str = "") -> None:
            self.words = {"the", "went", "home", "child", extra}

        def lookup(self, word: str) -> bool:
            return word.lower() in self.words

    ep = SimpleNamespace(name="mini", load=lambda: Words, dist=SimpleNamespace(name="p"))
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points",
        lambda group: [ep] if group.endswith(".spellfix-dictionary") else [],
    )
    fixed, corr = repair_text("The chi1d went home.", dict_name="mini", dict_options={"extra": "x"})
    assert fixed == "The child went home." and len(corr) == 1


def test_spellfix_stage_builds_dictionary_before_first_batch(tmp_path) -> None:
    from womblex.config import SpellfixConfig
    from womblex.process.spellfix_stage import spellfix_shards

    cfg = SpellfixConfig(dict_name="hunspell", dict_options={"name": "xx_XX"})
    with pytest.raises(FileNotFoundError, match="xx_XX"):
        spellfix_shards(tmp_path, cfg)


def test_paddleocr_reader_names_the_model_set_it_loaded() -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    from womblex.ingest.paddle_ocr import PaddleOCRReader

    reader = PaddleOCRReader()
    assert reader.model_variant is None
    reader._ensure_loaded()
    assert reader.model_variant in {"paddleocr-v5", "rapidocr-bundled-v4"}


# --- slot-model provenance (O3) --------------------------------------------

from womblex import __version__ as _WOMBLEX_VERSION


def test_resolve_alone_does_not_record_use(clean_registry) -> None:
    """Validation-only callers (``check_registered``, the early redaction and
    OCR-layout checks) must not read as though the model was built."""
    reg.reset_used_entries()
    reg.resolve(reg.SLOT_OCR, "paddleocr")
    assert reg.used_entries() == ()


def test_a_builtin_factory_call_records_womblex_as_the_distribution(
    clean_registry,
) -> None:
    reg.reset_used_entries()
    resolve_tokenizer("isaacus/kanon-2-tokenizer")
    (entry,) = reg.used_entries()
    assert (entry.slot, entry.name) == (reg.SLOT_TOKENIZER, "kanon-2-tokenizer")
    assert reg.distribution_version(entry) == ("womblex", _WOMBLEX_VERSION)


def test_a_plugin_factory_call_records_its_own_distribution_and_version(
    clean_registry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg.reset_used_entries()

    def make(**opts):
        return lambda text: len(text.split())

    ep = SimpleNamespace(
        name="words", load=lambda: make,
        dist=SimpleNamespace(name="my-pkg", version="1.2.3"),
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points", lambda group: [ep] if group.endswith(".tokenizer") else []
    )
    resolve_tokenizer("words")
    (entry,) = reg.used_entries()
    assert reg.distribution_version(entry) == ("my-pkg", "1.2.3")


def test_get_ocr_reader_records_its_entry(
    clean_registry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg.reset_used_entries()
    ep = SimpleNamespace(
        name="my-ocr", load=lambda: (lambda lang="eng", **o: "reader"),
        dist=SimpleNamespace(name="p", version="0.1"),
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [])
    get_ocr_reader(engine="my-ocr")
    assert {(e.slot, e.name) for e in reg.used_entries()} == {(reg.SLOT_OCR, "my-ocr")}


def test_get_layout_analyzer_caches_and_records_once(
    clean_registry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg.reset_used_entries()
    calls = []

    class Fake:
        def __init__(self, **opts):
            calls.append(opts)

    ep = SimpleNamespace(
        name="my-layout", load=lambda: Fake, dist=SimpleNamespace(name="p", version="2.0"),
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points", lambda group: [ep] if group.endswith(".layout") else []
    )
    get_layout_analyzer("my-layout", size=1)
    get_layout_analyzer("my-layout", size=1)
    assert len(calls) == 1
    assert len(reg.used_entries()) == 1


def test_pii_cleaner_records_on_first_use_not_construction(
    clean_registry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg.reset_used_entries()

    class Encoder:
        def __init__(self, **opts):
            pass

        def encode(self, texts):
            return np.ones((len(texts), 4))

    ep = SimpleNamespace(
        name="my-ctx", load=lambda: Encoder, dist=SimpleNamespace(name="p", version="3.0"),
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points", lambda group: [ep] if group.endswith(".pii-context") else []
    )
    cleaner = PIICleaner(model="my-ctx")
    assert reg.used_entries() == ()  # resolved eagerly, not yet built
    cleaner._load_model()
    (entry,) = reg.used_entries()
    assert reg.distribution_version(entry) == ("p", "3.0")


def test_spellfix_dictionary_records_its_entry(
    clean_registry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg.reset_used_entries()

    class Words:
        def lookup(self, word: str) -> bool:
            return True

    ep = SimpleNamespace(
        name="mini", load=lambda: Words, dist=SimpleNamespace(name="p", version="4.0"),
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points",
        lambda group: [ep] if group.endswith(".spellfix-dictionary") else [],
    )
    repair_text("The chi1d went home.", dict_name="mini")
    (entry,) = reg.used_entries()
    assert reg.distribution_version(entry) == ("p", "4.0")


def test_reset_used_entries_clears_the_record() -> None:
    reg.reset_used_entries()
    resolve_tokenizer("isaacus/kanon-2-tokenizer")
    assert reg.used_entries() != ()
    reg.reset_used_entries()
    assert reg.used_entries() == ()


# --- the default model group (O3) -------------------------------------------

from womblex.config import OCRConfig, PIIConfig, SpellfixConfig


def test_default_models_agree_with_each_slots_own_config_default() -> None:
    """DEFAULT_MODELS is held independently of config.py (no import), so a
    changed default in either place must not drift from the other unnoticed."""
    from womblex.ingest import paddle_ocr
    from womblex.pii import cleaner
    from womblex.process import chunker

    assert reg.resolve(reg.SLOT_OCR, OCRConfig().engine).name == reg.DEFAULT_MODELS[reg.SLOT_OCR]
    assert reg.resolve(
        reg.SLOT_LAYOUT, paddle_ocr.DEFAULT_LAYOUT_MODEL,
    ).name == reg.DEFAULT_MODELS[reg.SLOT_LAYOUT]
    assert reg.resolve(
        reg.SLOT_PII_CONTEXT, PIIConfig().model,
    ).name == reg.DEFAULT_MODELS[reg.SLOT_PII_CONTEXT]
    assert reg.resolve(
        reg.SLOT_TOKENIZER, chunker.DEFAULT_TOKENIZER,
    ).name == reg.DEFAULT_MODELS[reg.SLOT_TOKENIZER]
    assert reg.resolve(
        reg.SLOT_SPELLFIX_DICTIONARY, SpellfixConfig().dict_name,
    ).name == reg.DEFAULT_MODELS[reg.SLOT_SPELLFIX_DICTIONARY]
    assert cleaner.DEFAULT_CONTEXT_MODEL == PIIConfig().model


def test_is_default_group_true_for_the_defaults_by_name_or_alias() -> None:
    assert reg.is_default_group({reg.SLOT_OCR: "paddleocr", reg.SLOT_TOKENIZER: "kanon-2-tokenizer"})
    assert reg.is_default_group({reg.SLOT_OCR: "PaddleOCR"})  # alias, case-insensitive
    assert reg.is_default_group({})  # nothing named, nothing contradicted


def test_is_default_group_false_for_an_override() -> None:
    assert not reg.is_default_group({reg.SLOT_OCR: "mistral-ocr"})


def test_is_default_group_false_for_a_slot_the_default_group_does_not_cover() -> None:
    """Short-circuits before resolving, so an unregistered slot name does not
    need to exist for this to hold — a slot DEFAULT_MODELS does not cover
    cannot be "the default" for it, known or not."""
    assert not reg.is_default_group({"nonexistent-slot": "x"})
