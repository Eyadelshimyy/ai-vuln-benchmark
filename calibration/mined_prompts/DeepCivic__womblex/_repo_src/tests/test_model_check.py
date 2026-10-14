"""Pre-run model check: levels, scopes, failure reporting and what is remembered."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from womblex.config import WomblexConfig
from womblex.store.run_stamp import (
    MODEL_CHECK_KEY,
    RunStamp,
    read_footer_model_check,
)
from womblex.utils import model_registry as reg
from womblex.utils.model_check import (
    _BUILTIN_MODULES,
    SCOPE_CHUNK,
    SCOPE_EXTRACT,
    SCOPE_PII,
    SCOPE_SPELLFIX,
    CheckLevel,
    ModelCheckError,
    ModelCheckResult,
    SlotCheck,
    check_models,
    check_registered,
    configured_models,
    footer_payload,
    reset_model_check,
)

# The built-ins register at import; do it before the fixture copies the registry,
# or they would be registered into the copy and lost when it is restored.
for _module in _BUILTIN_MODULES:
    importlib.import_module(_module)

_PATHS = {"input_root": "/tmp/i", "output_root": "/tmp/o", "checkpoint_dir": "/tmp/c"}


def _config(**sections: dict) -> WomblexConfig:
    return WomblexConfig(dataset={"name": "t"}, paths=_PATHS, **sections)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    """Isolate registrations and what the process remembers having checked."""
    for attr in ("_entries", "_aliases"):
        monkeypatch.setattr(
            reg, attr, {slot: dict(v) for slot, v in getattr(reg, attr).items()}
        )
    reset_model_check()
    yield
    reset_model_check()


class _Dictionary:
    def __init__(self, knows: bool = True) -> None:
        self._knows = knows

    def lookup(self, word: str) -> bool:
        return self._knows


def _dictionary_config(name: str, **options: object) -> WomblexConfig:
    return _config(spellfix={"enabled": True, "dict_name": name, "dict_options": options})


class TestWhatTheConfigNames:
    def test_default_config_names_the_extraction_and_chunk_models(self) -> None:
        uses = {(u.slot, u.name): u.scopes for u in configured_models(_config())}
        assert uses[("ocr", "paddleocr")] == (SCOPE_EXTRACT,)
        # OCR and redaction share one analyser: listed once, loaded once.
        assert uses[("layout", "pp-doclayout-m")] == (SCOPE_EXTRACT,)
        assert uses[("tokenizer", "isaacus/kanon-2-tokenizer")] == (SCOPE_CHUNK,)

    def test_scopes_narrow_the_list(self) -> None:
        names = {u.slot for u in configured_models(_config(), scopes=(SCOPE_CHUNK,))}
        assert names == {"tokenizer"}

    def test_stages_that_are_off_are_not_checked(self) -> None:
        slots = {u.slot for u in configured_models(_config())}
        assert "spellfix-dictionary" not in slots
        assert "pii-context" not in slots

    def test_the_pii_model_is_named_only_when_the_backstop_uses_it(self) -> None:
        graph_only = _config(pii={"enabled": True})
        backstop = _config(pii={"enabled": True, "use_regex_backstop": True})
        assert not [u for u in configured_models(graph_only) if u.scopes == (SCOPE_PII,)]
        assert [u.name for u in configured_models(backstop, scopes=(SCOPE_PII,))] == [
            "all-minilm-l6-v2",
        ]

    def test_a_markdown_engine_with_no_redaction_has_no_layout_model(self) -> None:
        cfg = _config(
            extraction={"ocr": {"engine": "ollama"}},
            redaction={"enabled": False},
        )
        assert "layout" not in {u.slot for u in configured_models(cfg)}


class TestLevels:
    def test_off_checks_nothing_and_says_so(self) -> None:
        result = check_models(_config(), "off")
        assert result.checks == () and result.ok
        assert footer_payload() == {"checks": [], "level": "off"}

    def test_load_builds_the_model_through_its_factory(self) -> None:
        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "tiny", lambda **_: _Dictionary(), source="tiny-dist",
        )
        result = check_models(_dictionary_config("tiny"), "load", scopes=(SCOPE_SPELLFIX,))
        (check,) = result.checks
        assert (check.ok, check.name, check.source, check.level) == (
            True, "tiny", "tiny-dist", CheckLevel.LOAD,
        )

    def test_load_reports_a_factory_that_raises(self) -> None:
        def broken(**_: object) -> object:
            raise FileNotFoundError("weights missing")

        reg.register(reg.SLOT_SPELLFIX_DICTIONARY, "broken", broken, source="x")
        result = check_models(_dictionary_config("broken"), "load", scopes=(SCOPE_SPELLFIX,))
        (bad,) = result.failures
        assert "FileNotFoundError: weights missing" in bad.reason
        with pytest.raises(ModelCheckError, match="weights missing"):
            result.raise_for_failures()

    def test_smoke_catches_what_load_does_not(self) -> None:
        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "empty", lambda **_: _Dictionary(knows=False), source="x",
        )
        cfg = _dictionary_config("empty")
        assert check_models(cfg, "load", scopes=(SCOPE_SPELLFIX,)).failures == ()
        (bad,) = check_models(cfg, "smoke", scopes=(SCOPE_SPELLFIX,)).failures
        assert "the" in bad.reason

    def test_an_unknown_name_fails_and_lists_the_known_names(self) -> None:
        result = check_models(_dictionary_config("nope"), "load", scopes=(SCOPE_SPELLFIX,))
        (bad,) = result.failures
        assert "en_au" in bad.reason and "nope" in bad.reason

    def test_the_level_defaults_to_the_config(self) -> None:
        cfg = _config(processing={"models_check": "off"})
        assert check_models(cfg).level is CheckLevel.OFF


class TestVariant:
    def test_a_model_names_the_variant_it_resolved(self) -> None:
        class Reader:
            model_variant = "v5"

            def read_page(self, img: object) -> object:
                raise AssertionError("load must not infer")

        reg.register(reg.SLOT_OCR, "variant-ocr", lambda **_: Reader(), source="d")
        cfg = _config(extraction={"ocr": {"engine": "variant-ocr"}}, redaction={"enabled": False})
        result = check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        assert [(c.slot, c.variant) for c in result.checks if c.slot == "ocr"] == [("ocr", "v5")]

    def test_paddleocr_reports_which_model_set_loaded(self) -> None:
        pytest.importorskip("rapidocr_onnxruntime")
        cfg = _config(redaction={"enabled": False})
        result = check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        ocr = next(c for c in result.checks if c.slot == "ocr")
        assert ocr.ok, ocr.reason
        assert ocr.variant in {"paddleocr-v5", "rapidocr-bundled-v4"}

    def test_paddleocr_smoke_reads_the_probe_image(self) -> None:
        pytest.importorskip("rapidocr_onnxruntime")
        cfg = _config(redaction={"enabled": False})
        result = check_models(cfg, "smoke", scopes=(SCOPE_EXTRACT,))
        assert [c.reason for c in result.failures] == []


class TestWhatIsRemembered:
    def test_off_is_remembered_and_says_so(self) -> None:
        check_models(_config(), "off")
        assert footer_payload() == {"checks": [], "level": "off"}

    def test_nothing_asked_means_nothing_to_write(self) -> None:
        assert footer_payload() is None

    def test_the_footer_carries_the_checks_and_reads_back(self) -> None:
        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "tiny", lambda **_: _Dictionary(), source="tiny-dist",
        )
        check_models(_dictionary_config("tiny"), "load", scopes=(SCOPE_SPELLFIX,))
        meta = RunStamp.declare("run-1", _config(), stage="extract").footer_metadata()
        payload = read_footer_model_check(meta)
        assert payload is not None
        (entry,) = payload["checks"]
        assert (entry["slot"], entry["name"], entry["status"], entry["level"]) == (
            "spellfix-dictionary", "tiny", "ok", "load",
        )
        assert MODEL_CHECK_KEY.encode() in meta

    def test_an_unstamped_process_writes_no_key(self) -> None:
        meta = RunStamp.declare("run-1", _config(), stage="extract").footer_metadata()
        assert MODEL_CHECK_KEY.encode() not in meta

    def test_a_malformed_footer_reads_as_absent(self) -> None:
        assert read_footer_model_check({MODEL_CHECK_KEY.encode(): b"{not json"}) is None


class TestCheckDoesNotPollute:
    """Building a model to verify it loads is a probe, not the run using it:
    it must not land in the `womblex.slot_models` footer key, and the run's
    own later build of the same model must still land there. `load_dictionary`
    caches its build (so does `get_layout_analyzer`), so without bypassing
    that cache while the check is suppressed, the run's own build afterwards
    would be a cache hit that skips `record_use` entirely — never recorded.
    """

    def test_a_check_built_model_is_absent_from_the_slot_models_footer(self) -> None:
        from womblex.store.run_stamp import read_footer_slot_models

        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "quiet", lambda **_: _Dictionary(), source="quiet-dist",
        )
        cfg = _dictionary_config("quiet")
        check_models(cfg, "load", scopes=(SCOPE_SPELLFIX,))
        meta = RunStamp.declare("run-1", cfg, stage="spellfix").footer_metadata()
        assert "quiet" not in {e["name"] for e in read_footer_slot_models(meta)}

    def test_the_runs_own_build_afterwards_is_present_in_the_footer(self) -> None:
        from womblex.process.spellfix import load_dictionary
        from womblex.store.run_stamp import read_footer_slot_models

        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "quiet2", lambda **_: _Dictionary(), source="quiet-dist",
        )
        cfg = _dictionary_config("quiet2")
        check_models(cfg, "load", scopes=(SCOPE_SPELLFIX,))
        load_dictionary("quiet2")
        meta = RunStamp.declare("run-1", cfg, stage="spellfix").footer_metadata()
        assert "quiet2" in {e["name"] for e in read_footer_slot_models(meta)}

    def test_a_check_built_layout_model_is_absent_from_what_this_process_used(self) -> None:
        reg.register(reg.SLOT_OCR, "quiet-ocr", lambda **_: object(), source="quiet-dist")
        reg.register(reg.SLOT_LAYOUT, "quiet-layout", lambda **_: object(), source="quiet-dist")
        cfg = _config(
            extraction={"ocr": {"engine": "quiet-ocr"}}, layout={"model": "quiet-layout"},
            redaction={"enabled": False},
        )
        before = {e.name for e in reg.used_entries()}
        check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        assert {e.name for e in reg.used_entries()} == before

    def test_the_runs_own_layout_build_afterwards_is_still_recorded(self) -> None:
        from womblex.ingest.paddle_ocr import get_layout_analyzer

        reg.register(reg.SLOT_OCR, "quiet-ocr-2", lambda **_: object(), source="quiet-dist")
        reg.register(reg.SLOT_LAYOUT, "quiet-layout-2", lambda **_: object(), source="quiet-dist")
        cfg = _config(
            extraction={"ocr": {"engine": "quiet-ocr-2"}}, layout={"model": "quiet-layout-2"},
            redaction={"enabled": False},
        )
        check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        assert "quiet-layout-2" not in {e.name for e in reg.used_entries()}
        get_layout_analyzer("quiet-layout-2")
        assert "quiet-layout-2" in {e.name for e in reg.used_entries()}

    def test_the_runs_own_dictionary_artefact_is_still_recorded(self) -> None:
        # The artefact (`womblex.models`) side: the bundled dictionary's parse
        # is cached, so its resolution must not sit behind that cache.
        pytest.importorskip("spylls")
        from womblex.process import spellfix
        from womblex.utils.models import loaded_models, reset_loaded_models

        spellfix._cached_dictionary.cache_clear()
        reset_loaded_models()
        check_models(_dictionary_config("en_AU"), "load", scopes=(SCOPE_SPELLFIX,))
        assert "en_AU" not in {m.name for m in loaded_models()}
        spellfix.load_dictionary("en_AU")
        assert "en_AU" in {m.name for m in loaded_models()}


def _failed(slot: str, scope: str) -> ModelCheckResult:
    check = SlotCheck(slot, "gone", (scope,), CheckLevel.LOAD, ok=False, reason="not found")
    return ModelCheckResult(CheckLevel.LOAD, (check,))


class TestRunStopsBeforeTheFirstDocument:
    def test_a_failed_check_writes_nothing(self, tmp_path: Path) -> None:
        import argparse

        from womblex.cli.pipeline import cmd_run

        inbox = tmp_path / "in"
        inbox.mkdir()
        (inbox / "a.csv").write_text("a,b\n1,2\n")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            f"dataset:\n  name: t\npaths:\n  input_root: {inbox}\n"
            f"  output_root: {tmp_path / 'out'}\n  checkpoint_dir: {tmp_path / 'ckpt'}\n"
            "extraction:\n  ocr:\n    engine: no-such-engine\n"
            "redaction:\n  enabled: false\n"
        )
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0, batch_size=None, run_id="t",
            models_check=None,
        )
        assert cmd_run(args) == 1
        assert not (tmp_path / "out").exists()

    def test_a_later_stage_model_also_stops_the_run(self, tmp_path: Path) -> None:
        import argparse

        from womblex.cli.pipeline import cmd_run

        inbox = tmp_path / "in"
        inbox.mkdir()
        (inbox / "a.csv").write_text("a,b\n1,2\n")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            f"dataset:\n  name: t\npaths:\n  input_root: {inbox}\n"
            f"  output_root: {tmp_path / 'out'}\n  checkpoint_dir: {tmp_path / 'ckpt'}\n"
            "redaction:\n  enabled: false\n"
            "spellfix:\n  enabled: true\n  dict_name: no-such-dictionary\n"
        )
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0, batch_size=None, run_id="t",
            models_check=None,
        )
        assert cmd_run(args) == 1
        assert not (tmp_path / "out").exists()

    def test_the_cli_override_beats_the_config(self, tmp_path: Path) -> None:
        import argparse

        from womblex.cli._shared import apply_models_check

        cfg = _config()
        apply_models_check(cfg, argparse.Namespace(models_check="smoke"))
        assert cfg.processing.models_check == "smoke"
        apply_models_check(cfg, argparse.Namespace(models_check=None))
        assert cfg.processing.models_check == "smoke"



class TestWorkerRefusal:
    def test_a_queue_is_joined_only_if_its_models_pass(self) -> None:
        from womblex.cloud import dbos_app
        from womblex.cloud.worker import served_queues

        root = "s3://bucket/inbox"
        extraction = dbos_app.extract_queue(root)
        chunk, money = dbos_app.stage_queue("chunk"), dbos_app.stage_queue("money")

        extraction_down = served_queues(_failed("ocr", SCOPE_EXTRACT), root)
        assert extraction not in extraction_down
        assert chunk in extraction_down and money in extraction_down

        tokenizer_down = served_queues(_failed("tokenizer", SCOPE_CHUNK), root)
        assert extraction in tokenizer_down
        assert chunk not in tokenizer_down and money in tokenizer_down

    def test_coordinators_are_always_served(self) -> None:
        from womblex.cloud import dbos_app
        from womblex.cloud.worker import served_queues

        queues = served_queues(_failed("ocr", SCOPE_EXTRACT), "s3://bucket/inbox")
        assert dbos_app.DOWNSTREAM_QUEUE in queues


class TestStagePreflight:
    def test_a_stage_whose_model_fails_is_refused_before_it_runs(self) -> None:
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS
        from womblex.cloud.stage_runner import StagePreconditionError, prepare_stage_context

        cfg = _config(chunking={"tokenizer": "nope"})
        with pytest.raises(StagePreconditionError, match="chunk model check failed.*nope"):
            prepare_stage_context(STAGE_CONTRACTS["chunk"], cfg)

    def test_a_stage_without_a_model_is_unaffected(self) -> None:
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS
        from womblex.cloud.stage_runner import prepare_stage_context

        cfg = _config(chunking={"tokenizer": "nope"})
        assert prepare_stage_context(STAGE_CONTRACTS["money"], cfg) is not None

    def test_off_skips_the_check(self) -> None:
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS
        from womblex.cloud.stage_runner import prepare_stage_context

        cfg = _config(chunking={"tokenizer": "nope"}, processing={"models_check": "off"})
        assert prepare_stage_context(STAGE_CONTRACTS["chunk"], cfg) is not None


class TestRegistrationOnly:
    def test_a_registered_model_is_not_loaded(self) -> None:
        def factory(**_: object) -> object:
            raise AssertionError("registration check must not call the factory")

        reg.register(reg.SLOT_SPELLFIX_DICTIONARY, "unloaded", factory, source="x")
        check_registered(_dictionary_config("unloaded"))

    def test_an_unknown_model_lists_the_known_names(self) -> None:
        with pytest.raises(ValueError, match="en_au"):
            check_registered(_dictionary_config("nope"))


class TestSubmission:
    def test_an_unknown_model_is_rejected_before_any_row_is_written(self, tmp_path: Path) -> None:
        from womblex.cloud import dispatch
        from womblex.ui.deps import UISettings

        settings = UISettings(output_root=tmp_path, store_uri=None)
        with pytest.raises(ValueError, match="nope.*known"):
            dispatch.downstream_stages(settings, {"chunking": {"tokenizer": "nope"}})

    def test_registered_models_pass_without_being_loaded(self, tmp_path: Path) -> None:
        from womblex.cloud import dispatch
        from womblex.ui.deps import UISettings

        settings = UISettings(output_root=tmp_path, store_uri=None)
        assert isinstance(dispatch.downstream_stages(settings, {}), tuple)


class TestStandaloneStageCommands:
    """`chunk`, `pii`, `spellfix` and `redact` check their models before writing anything."""

    @staticmethod
    def _run(register, command, *argv: str) -> int:
        import argparse

        parser = argparse.ArgumentParser()
        register(parser)
        return command(parser.parse_args(list(argv)))

    @staticmethod
    def _shards(tmp_path: Path, *names: str) -> Path:
        shards = tmp_path / "documents"
        shards.mkdir()
        for name in names:
            (shards / name).touch()
        return shards

    @staticmethod
    def _config(tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "dataset:\n  name: t\npaths:\n  input_root: .\n  output_root: .\n"
            f"  checkpoint_dir: .\n{body}"
        )
        return cfg

    def test_chunk_stops_on_an_unknown_tokeniser(self, tmp_path: Path) -> None:
        from womblex.cli.pipeline import _register_chunk, cmd_chunk

        shards = self._shards(tmp_path, "batch-0001._manifest.parquet")
        cfg = self._config(tmp_path, "chunking:\n  tokenizer: nope\n")
        rc = self._run(_register_chunk, cmd_chunk, "--shards", str(shards), "--config", str(cfg))
        assert rc == 1
        assert not (tmp_path / ".chunk-checkpoint").exists()

    def test_pii_stops_on_an_unknown_context_model(self, tmp_path: Path) -> None:
        from womblex.cli.pii import _register_pii, cmd_pii

        shards = self._shards(tmp_path, "batch-0001.chunks.parquet")
        cfg = self._config(tmp_path, "pii:\n  use_regex_backstop: true\n  model: nope\n")
        rc = self._run(_register_pii, cmd_pii, "--shards", str(shards), "--config", str(cfg))
        assert rc == 1
        assert not (tmp_path / ".pii-checkpoint").exists()

    def test_spellfix_stops_on_an_unknown_dictionary(self, tmp_path: Path) -> None:
        from womblex.cli.spellfix import _register_spellfix, cmd_spellfix

        shards = self._shards(tmp_path, "batch-0001.elements.parquet")
        cfg = self._config(tmp_path, "spellfix:\n  dict_name: nope\n")
        rc = self._run(
            _register_spellfix, cmd_spellfix, "--shards", str(shards), "--config", str(cfg),
        )
        assert rc == 1
        assert not (tmp_path / ".spellfix-checkpoint").exists()

    def test_the_flag_switches_the_check_off(self) -> None:
        import argparse

        from womblex.cli._shared import check_stage_models
        from womblex.config import ChunkingConfig

        bad = ChunkingConfig(tokenizer="nope")
        assert not check_stage_models(argparse.Namespace(models_check=None), (SCOPE_CHUNK,), chunking=bad)
        assert check_stage_models(argparse.Namespace(models_check="off"), (SCOPE_CHUNK,), chunking=bad)


class TestMarkdownEngines:
    """An engine that returns page markdown (as Mistral and Ollama do) is smoke-tested too."""

    @staticmethod
    def _register(name: str, markdown: str | None) -> None:
        from womblex.ingest.interfaces.protocols import OCRPageResult

        class Reader:
            def read_page(self, img: object) -> OCRPageResult:
                return OCRPageResult(markdown=markdown, reading_order_native=True, confidence=1.0)

        reg.register(reg.SLOT_OCR, name, lambda **_: Reader(), traits={"markdown": True}, source="d")

    def _cfg(self, name: str) -> WomblexConfig:
        return _config(extraction={"ocr": {"engine": name}}, redaction={"enabled": False})

    def test_it_passes_when_it_reads_the_probe(self) -> None:
        self._register("md-ok", "Womblex model check")
        result = check_models(self._cfg("md-ok"), "smoke", scopes=(SCOPE_EXTRACT,))
        assert [c.slot for c in result.checks] == ["ocr"] and result.ok

    def test_an_engine_that_swallows_its_errors_still_fails_smoke(self) -> None:
        # Bedrock and Ollama readers log a failed request and return empty
        # markdown at confidence 0; that must not read as a healthy engine.
        self._register("md-empty", None)
        result = check_models(self._cfg("md-empty"), "smoke", scopes=(SCOPE_EXTRACT,))
        assert "no text" in result.failures[0].reason

    def test_load_alone_does_not_call_the_service(self) -> None:
        self._register("md-empty", None)
        assert check_models(self._cfg("md-empty"), "load", scopes=(SCOPE_EXTRACT,)).ok


# --- models behind a service: one minimal request before any document -------

from types import SimpleNamespace

from womblex.ingest.llm_ocr import MistralOCRReader, OllamaOCRReader
from womblex.utils import isaacus_client
from womblex.utils.model_check import ISAACUS, SCOPE_EMBED, SCOPE_ENRICH, isaacus_models


class _FakeIsaacus:
    """Records what was called; *fail* makes every request raise."""

    def __init__(self, fail: str | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

        def call(kind: str, result: object):
            def create(**kwargs: object) -> object:
                self.calls.append((kind, kwargs))
                if fail:
                    raise RuntimeError(fail)
                return result
            return create

        self.embeddings = SimpleNamespace(create=call(
            "embed", SimpleNamespace(embeddings=[SimpleNamespace(embedding=[0.1, 0.2])]),
        ))
        self.enrichments = SimpleNamespace(create=call(
            "enrich", SimpleNamespace(results=[object()]),
        ))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_isaacus(monkeypatch: pytest.MonkeyPatch):
    made: list[_FakeIsaacus] = []

    def install(fail: str | None = None) -> list[_FakeIsaacus]:
        def make(*, models: object = ()) -> _FakeIsaacus:
            made.append(_FakeIsaacus(fail))
            return made[-1]

        monkeypatch.setattr(isaacus_client, "make_isaacus_client", make)
        return made

    return install


_PAID = {
    "enrichment": {"enabled": True},
    "embedding": {"enabled": True},
    "chunking": {"chunking_model": "kanon-2-enricher"},
}


class TestIsaacus:
    def test_each_enabled_stage_names_the_model_it_calls(self) -> None:
        uses = {u.name: u.scopes for u in isaacus_models(_config(**_PAID))}
        # AI chunking and enrich call the same enricher: one check, both scopes.
        assert uses == {
            "kanon-2-enricher": (SCOPE_ENRICH, SCOPE_CHUNK),
            "kanon-2-embedder": (SCOPE_EMBED,),
        }

    def test_nothing_is_called_when_no_stage_needs_isaacus(self) -> None:
        # Plain chunking runs offline on the vendored tokeniser.
        assert isaacus_models(_config()) == []

    def test_the_run_check_pings_every_model_and_records_it(self, fake_isaacus) -> None:
        made = fake_isaacus()
        result = check_models(_config(**_PAID), "load", scopes=(SCOPE_ENRICH, SCOPE_EMBED))
        assert result.ok
        kinds = sorted(c.calls[0][0] for c in made)
        assert kinds == ["embed", "enrich"]
        embed = next(c for c in made if c.calls[0][0] == "embed")
        assert embed.calls[0][1]["task"] == "retrieval/document"
        assert all(c.closed for c in made)
        assert {(c.slot, c.source) for c in result.checks} == {(ISAACUS, "isaacus-api")}
        names = {e["name"] for e in footer_payload()["checks"]}
        assert names == {"kanon-2-enricher", "kanon-2-embedder"}

    def test_a_failure_stops_the_stage_that_needs_it_and_only_that_stage(
        self, fake_isaacus,
    ) -> None:
        fake_isaacus(fail="401 invalid API key")
        cfg = _config(enrichment={"enabled": True}, spellfix={"enabled": True})
        result = check_models(cfg, "load", scopes=(SCOPE_ENRICH,))
        (bad,) = result.failures
        assert "401 invalid API key" in bad.reason
        assert result.failures_for((SCOPE_ENRICH,)) and not result.failures_for((SCOPE_EXTRACT,))

    def test_womblex_run_checks_isaacus_before_extraction(self) -> None:
        """No scopes = every enabled stage, which is what `womblex run` asks for."""
        cfg = _config(embedding={"enabled": True})
        assert [(u.name, u.scopes) for u in isaacus_models(cfg)] == [
            ("kanon-2-embedder", (SCOPE_EMBED,)),
        ]

    def test_registration_only_never_calls_the_service(self, fake_isaacus) -> None:
        made = fake_isaacus()
        check_registered(_config(**_PAID))
        assert made == []


class TestPing:
    def test_a_registered_model_with_ping_is_pinged_at_load(self) -> None:
        pinged: list[bool] = []

        class Remote:
            def ping(self) -> None:
                pinged.append(True)
                raise PermissionError("model access not granted")

        reg.register(reg.SLOT_OCR, "remote-ocr", lambda **_: Remote(), source="d")
        cfg = _config(extraction={"ocr": {"engine": "remote-ocr"}}, redaction={"enabled": False})
        result = check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        (bad,) = [c for c in result.failures if c.slot == "ocr"]
        assert pinged and "model access not granted" in bad.reason

    def test_mistral_ping_is_a_one_token_request(self) -> None:
        sent: list[dict] = []
        reader = MistralOCRReader(model="m", region="ap-southeast-2")
        reader._client = SimpleNamespace(converse=lambda **kw: sent.append(kw))
        reader.ping()
        assert sent[0]["modelId"] == "m"
        assert sent[0]["inferenceConfig"] == {"maxTokens": 1}

    def test_mistral_ping_raises_what_bedrock_raises(self) -> None:
        def denied(**_: object) -> None:
            raise RuntimeError("AccessDeniedException")

        reader = MistralOCRReader(model="m")
        reader._client = SimpleNamespace(converse=denied)
        with pytest.raises(RuntimeError, match="AccessDenied"):
            reader.ping()

    @pytest.mark.parametrize("served,ok", [
        (["llama3.2-vision:latest"], True),
        (["llama3.2-vision"], True),
        (["qwen2.5vl:7b"], False),
    ])
    def test_ollama_ping_needs_the_model_pulled(self, served: list[str], ok: bool) -> None:
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": s} for s in served]},
        )
        reader = OllamaOCRReader(model="llama3.2-vision", base_url="http://h:1")
        reader._client = SimpleNamespace(get=lambda url: response, close=lambda: None)
        if ok:
            reader.ping()
        else:
            with pytest.raises(RuntimeError, match="ollama pull"):
                reader.ping()
