"""The run record: what produced a run, observed from its own artefacts.

One case per acceptance criterion the mechanism can decide on its own — where
the record lives, that it regenerates from the shard directory alone and
reproduces apart from its timestamp, that stages are observed rather than
declared, that a model digest recomputes from the model files, that no
credential reaches the footer, and that a run predating the stamps is marked
partial rather than failing.

Shards are written through the real writer on the synthetic budget-statement
DOCX, as `test_run_stamp` does.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._synthetic import BUDGET_DOCX as _BUDGET_DOCX
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.build_info import IMAGE_REF_ENV, image_info
from womblex.store.embed_output import EMBEDDINGS_SCHEMA
from womblex.store.output import read_manifest, write_results
from womblex.store.run_manifest import (
    RUN_RECORD_KEY,
    RUN_RECORD_VERSION,
    _footers,
    build_run_record,
    read_run_record,
    run_manifest_path_for,
    write_run_manifest,
)
from womblex.store.run_stamp import RunStamp
from womblex.utils import model_registry as reg
from womblex.utils.model_check import (
    CheckLevel,
    SlotCheck,
    check_models,
    reset_model_check,
)
from womblex.utils.models import (
    digest_model_path,
    reset_loaded_models,
    resolve_local_model_path,
)

# The names a run's credentials arrive under; none of their values may reach
# the record, which carries endpoints, regions, digests and model ids.
CREDENTIAL_ENV = (
    "ISAACUS_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "WOMBLEX_S3_ACCESS_KEY_ID",
    "WOMBLEX_S3_SECRET_ACCESS_KEY",
    "WOMBLEX_DB_DSN",
)


@pytest.fixture(autouse=True)
def _clean_record():
    reset_loaded_models()
    reset_model_check()
    reg.reset_used_entries()
    yield
    reset_loaded_models()
    reset_model_check()
    reg.reset_used_entries()


@pytest.fixture(scope="module")
def extraction():
    return DocxExtractor().extract_path(_BUDGET_DOCX)


def _config(tmp_path: Path) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="test-corpus"),
        paths=PathsConfig(
            input_root=tmp_path / "in",
            output_root=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
        ),
    )


def _extracted(tmp_path: Path, extraction, *, stamp: RunStamp | None = None) -> Path:
    """A one-batch shard directory, stamped for ``run-A`` unless told otherwise."""
    shards = tmp_path / "run-A" / "documents"
    write_results(
        [("budget", str(_BUDGET_DOCX), extraction)],
        shards / "batch-0001.parquet",
        collection_id="test-corpus",
        stamp=stamp if stamp is not None else RunStamp.declare(
            "run-A", _config(tmp_path), stage="extract",
        ),
    )
    return shards


def _sidecar(shards: Path, suffix: str, table: pa.Table, stage: str) -> Path:
    """A downstream sidecar of batch 1, stamped as written by *stage*."""
    target = shards / f"batch-0001{suffix}"
    stamp = RunStamp.inherit("run-A", "sha256:x", stage=stage)
    pq.write_table(
        table.replace_schema_metadata(stamp.footer_metadata()), str(target),
        compression="zstd",
    )
    return target


class TestWhereItLives:
    def test_the_record_is_footer_metadata_on_the_run_manifest(self, tmp_path, extraction):
        """No new output file, no new column: the manifest's own footer."""
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(shards)

        assert manifest == run_manifest_path_for(shards)
        assert sorted(p.name for p in manifest.parent.iterdir()) == [
            "documents", "manifest.parquet",
        ]
        table = pq.read_table(str(manifest))
        assert RUN_RECORD_KEY not in table.schema.names
        assert read_run_record(manifest)["record_version"] == RUN_RECORD_VERSION

    def test_a_reader_ignoring_the_footer_reads_the_manifest_unchanged(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(shards)
        assert pq.read_table(str(manifest)).num_rows == 1


class TestFileChecksums:
    def test_supplied_checksums_are_listed_sorted(self, tmp_path, extraction):
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(
            shards, checksums={"documents/b.parquet": "bb", "documents/a.parquet": "aa"},
        )
        record = read_run_record(manifest)
        assert record["files"] == [
            {"key": "documents/a.parquet", "sha256": "aa"},
            {"key": "documents/b.parquet", "sha256": "bb"},
        ]
        assert not any("file checksums" in gap for gap in record["partial"])

    def test_jobs_that_may_have_published_undigested_files_are_named(
        self, tmp_path, extraction,
    ):
        manifest = write_run_manifest(
            _extracted(tmp_path, extraction),
            checksums={"documents/a.parquet": "aa"}, undigested=["r1:stage:chunk"],
        )
        gaps = [g for g in read_run_record(manifest)["partial"] if "file checksums" in g]
        assert len(gaps) == 1 and "r1:stage:chunk" in gaps[0]

    def test_a_record_without_them_names_the_gap(self, tmp_path, extraction):
        record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
        assert "files" not in record
        assert any("file checksums" in gap for gap in record["partial"])


class TestRegeneration:
    def test_the_shard_directory_alone_regenerates_the_record(self, tmp_path, extraction):
        """No configuration is consulted — `womblex manifest` gets a directory."""
        shards = _extracted(tmp_path, extraction)
        first = read_run_record(write_run_manifest(shards))
        assert first["run"]["run_id"] == "run-A"
        assert first["inputs"]["documents"] == 1
        assert first["inputs"]["collection_id"] == "test-corpus"

    def test_re_running_over_an_unchanged_run_reproduces_the_record(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        first = read_run_record(write_run_manifest(shards))
        second = read_run_record(write_run_manifest(shards))
        assert first.pop("generated_at") is not None
        second.pop("generated_at")
        assert first == second


class TestStagesAreObserved:
    def test_a_stage_is_named_by_the_files_it_wrote(self, tmp_path, extraction):
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        stages = {s["stage"]: s for s in record["stages"]}
        assert set(stages) == {"extract"}
        assert stages["extract"]["files"] == 4
        assert stages["extract"]["rows"] > 0

    def test_a_stage_that_ran_and_produced_nothing_differs_from_one_that_never_ran(
        self, tmp_path, extraction,
    ):
        """The distinction a list taken from configuration flags cannot make:
        present with no rows, versus absent."""
        shards = _extracted(tmp_path, extraction)
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
            schema=EMBEDDINGS_SCHEMA,
        )
        _sidecar(shards, ".embeddings.parquet", empty, "embed")

        stages = {s["stage"]: s for s in read_run_record(write_run_manifest(shards))["stages"]}
        assert stages["embed"] == {"stage": "embed", "files": 1, "rows": 0}
        assert "chunk" not in stages

    def test_stages_come_back_in_pipeline_order(self, tmp_path, extraction):
        shards = _extracted(tmp_path, extraction)
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
            schema=EMBEDDINGS_SCHEMA,
        )
        _sidecar(shards, ".embeddings.parquet", empty, "embed")
        names = [s["stage"] for s in read_run_record(write_run_manifest(shards))["stages"]]
        assert names == ["extract", "embed"]


class TestLocalModels:
    def test_a_loaded_model_is_named_with_a_digest_that_recomputes(
        self, tmp_path, extraction,
    ):
        """The criterion: the digest checks out against the model files alone."""
        resolve_local_model_path("en_AU")
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))

        entry = next(m for m in record["local_models"] if m["name"] == "en_AU")
        assert entry["digest"] == digest_model_path(
            Path(str(resolve_local_model_path("en_AU"))),
        )
        assert entry["stages"] == ["extract"]

    def test_a_run_that_loaded_none_says_so_rather_than_listing_the_build(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        assert record["local_models"] == []
        assert any("local models" in p for p in record["partial"])

    def test_a_plugin_weight_appears_alongside_a_builtin_artefact(
        self, tmp_path, extraction, monkeypatch: pytest.MonkeyPatch,
    ):
        """A plugin that loads its own bundled weights (no models root
        involved) calls `record_loaded_path` directly, the way the built-in
        PaddleOCR v4 fallback does, and lands in the same record (O3)."""
        weights = tmp_path / "plugin-weights"
        weights.mkdir()
        (weights / "model.bin").write_bytes(b"plugin weights")

        def make_reader(**opts):
            from womblex.utils.models import record_loaded_path

            record_loaded_path("my-ocr-weights", weights)
            return object()

        ep = SimpleNamespace(
            name="my-ocr", load=lambda: make_reader,
            dist=SimpleNamespace(name="my-pkg", version="1.0"),
        )
        monkeypatch.setattr(reg, "_loaded", set())
        monkeypatch.setattr(
            reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [],
        )

        resolve_local_model_path("en_AU")  # a built-in artefact
        from womblex.ingest.paddle_ocr import get_ocr_reader

        get_ocr_reader(engine="my-ocr")  # the plugin's own bundled artefact

        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        names = {m["name"] for m in record["local_models"]}
        assert names == {"en_AU", "my-ocr-weights"}


class TestSlotModels:
    """`slot_models`: which swappable-slot model each stage actually built,
    by distribution and version (O3) — distinct from `local_models`, which
    names model *artefacts* by digest.
    """

    def test_a_built_model_is_named_by_slot_distribution_and_version(
        self, tmp_path, extraction,
    ):
        from womblex import __version__
        from womblex.process.chunker import resolve_tokenizer

        resolve_tokenizer("isaacus/kanon-2-tokenizer")
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))

        entry = next(m for m in record["slot_models"] if m["slot"] == "tokenizer")
        assert entry == {
            "slot": "tokenizer", "name": "kanon-2-tokenizer",
            "distribution": "womblex", "version": __version__, "stages": ["extract"],
        }

    def test_a_run_that_built_none_says_so_rather_than_listing_the_build(
        self, tmp_path, extraction,
    ):
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        assert record["slot_models"] == []
        assert any("slot models" in p for p in record["partial"])

    def test_the_ocr_engine_gap_closes_once_a_slot_entry_names_one(
        self, tmp_path, extraction,
    ):
        """The criterion: the run record no longer lists the OCR engine as
        unestablished, once `slot_models` carries an `ocr` entry."""
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        assert any("OCR engine is not recorded" in p for p in record["partial"])

        from womblex.ingest.paddle_ocr import get_ocr_reader

        get_ocr_reader(engine="mistral-ocr")
        shards = _extracted(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        assert not any("OCR engine is not recorded" in p for p in record["partial"])
        assert any(m["slot"] == "ocr" for m in record["slot_models"])


class TestCredentials:
    def test_no_credential_value_reaches_the_record(self, tmp_path, extraction, monkeypatch):
        for name in CREDENTIAL_ENV:
            monkeypatch.setenv(name, f"SECRET-{name}")
        monkeypatch.setenv(
            "ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2=kanon-2-embedder",
        )
        shards = _extracted(tmp_path, extraction)
        manifest = write_run_manifest(shards)

        blob = pq.read_metadata(str(manifest)).metadata[RUN_RECORD_KEY.encode()]
        for name in CREDENTIAL_ENV:
            assert f"SECRET-{name}".encode() not in blob

    def test_a_deployment_is_named_by_endpoint_region_and_models(
        self, tmp_path, extraction, monkeypatch,
    ):
        monkeypatch.setenv(
            "ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2=kanon-2-embedder",
        )
        shards = _extracted(tmp_path, extraction)
        assert read_run_record(write_run_manifest(shards))["services"] == [{
            "kind": "isaacus-sagemaker",
            "endpoint": "embed-001",
            "region": "ap-southeast-2",
            "models": ["kanon-2-embedder"],
        }]


class TestPartialRecord:
    def test_a_run_predating_the_stamps_is_marked_partial_not_failed(
        self, tmp_path, extraction,
    ):
        """An unstamped run still yields documents, methods and statuses; the
        one fact it cannot supply is named rather than invented."""
        shards = _extracted(tmp_path, extraction, stamp=None)
        for path in shards.glob("*.parquet"):
            table = pq.read_table(str(path))
            pq.write_table(table.replace_schema_metadata({}), str(path))

        record = read_run_record(write_run_manifest(shards))
        assert record["run"] is None
        assert record["stages"] == []
        assert record["inputs"]["documents"] == 1
        assert any("run identity" in p for p in record["partial"])
        assert any("stages" in p for p in record["partial"])

    def test_an_empty_shard_directory_records_the_absence_rather_than_raising(
        self, tmp_path,
    ):
        shards = tmp_path / "run-A" / "documents"
        shards.mkdir(parents=True)
        record = read_run_record(write_run_manifest(shards))
        assert record["inputs"]["documents"] == 0
        assert any("no readable parquet" in p for p in record["partial"])

class TestStagedSubset:
    """The distributed shape: `womblex finalize` stages in only the manifests.

    A record built from that directory can see extraction and nothing else, and
    a missing stage is indistinguishable from one that never ran — so it either
    says so, or is handed the footers read where the shards actually live.
    """

    @pytest.fixture
    def with_embed(self, tmp_path, extraction) -> Path:
        shards = _extracted(tmp_path, extraction)
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
            schema=EMBEDDINGS_SCHEMA,
        )
        _sidecar(shards, ".embeddings.parquet", empty, "embed")
        return shards

    @staticmethod
    def _manifests_only(shards: Path, tmp_path: Path) -> Path:
        staged = tmp_path / "staged" / "documents"
        staged.mkdir(parents=True)
        for path in shards.glob("*._manifest.parquet"):
            shutil.copy(path, staged / path.name)
        return staged

    def test_a_manifest_only_directory_declares_what_it_cannot_see(
        self, with_embed, tmp_path,
    ):
        staged = self._manifests_only(with_embed, tmp_path)
        record = read_run_record(write_run_manifest(staged))
        assert [s["stage"] for s in record["stages"]] == ["extract"]
        assert any("manifest shards alone" in p for p in record["partial"])

    def test_supplied_footers_restore_the_stages_the_staging_left_behind(
        self, with_embed, tmp_path,
    ):
        """What `womblex finalize` does: read the footers where the shards live
        and hand them in, rather than staging the whole run to observe it."""
        staged = self._manifests_only(with_embed, tmp_path)
        record = read_run_record(
            write_run_manifest(staged, footers=_footers(with_embed)),
        )
        assert [s["stage"] for s in record["stages"]] == ["extract", "embed"]
        assert not any("manifest shards alone" in p for p in record["partial"])

    def test_a_full_shard_directory_makes_no_such_claim(self, with_embed):
        record = read_run_record(write_run_manifest(with_embed))
        assert not any("manifest shards alone" in p for p in record["partial"])
        assert not any("does not hold the whole run" in p for p in record["partial"])

    def test_supplied_footers_still_declare_the_served_model_limit(
        self, with_embed, tmp_path,
    ):
        """Footers carry the stages and models, but served models are read from
        the embedding sidecars' rows — which supplied footers cannot stand in
        for, so the limit outlives the fix for the other two."""
        staged = self._manifests_only(with_embed, tmp_path)
        record = read_run_record(
            write_run_manifest(staged, footers=_footers(with_embed)),
        )
        assert any("does not hold the whole run" in p for p in record["partial"])


class TestEndpointIdentity:
    """No account identifier reaches the record, whatever shape the ARN takes."""

    ARNS = (
        "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/embed-001",
        "arn:aws:sagemaker:ap-southeast-2:123456789012",
        "arn:aws:sagemaker:ap-southeast-2:123456789012:",
        "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/",
        "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/a/embed-001",
    )

    @pytest.mark.parametrize("arn", ARNS)
    def test_no_arn_shape_puts_the_account_id_in_the_record(
        self, arn, tmp_path, extraction, monkeypatch,
    ):
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", f"{arn}=kanon-2-embedder")
        shards = _extracted(tmp_path, extraction)
        blob = pq.read_metadata(
            str(write_run_manifest(shards)),
        ).metadata[RUN_RECORD_KEY.encode()]
        assert b"123456789012" not in blob

    def test_an_arn_is_reduced_to_its_endpoint_name_and_region(
        self, tmp_path, extraction, monkeypatch,
    ):
        monkeypatch.setenv(
            "ISAACUS_SAGEMAKER_ENDPOINTS",
            "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/embed-001"
            "=kanon-2-embedder",
        )
        shards = _extracted(tmp_path, extraction)
        assert read_run_record(write_run_manifest(shards))["services"] == [{
            "kind": "isaacus-sagemaker",
            "endpoint": "embed-001",
            "region": "ap-southeast-2",
            "models": ["kanon-2-embedder"],
        }]

    def test_a_plain_endpoint_name_is_untouched(self, tmp_path, extraction, monkeypatch):
        monkeypatch.setenv("ISAACUS_SAGEMAKER_ENDPOINTS", "embed-001@ap-southeast-2")
        services = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
        assert services["services"][0]["endpoint"] == "embed-001"


class TestTheImageTheRunExecutedInside:
    """R4: the record names the image by digest, or says honestly why it cannot.

    The digest has to be handed in — it is content-addressed after the push, so
    it cannot be baked into the image it names, and a container cannot read its
    own labels. So every one of these outcomes is a *recorded* fact and none is
    an empty value standing in for one.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        image_info.cache_clear()
        yield
        image_info.cache_clear()

    @staticmethod
    def _in_container(monkeypatch, ref: str | None) -> None:
        """Run as though inside a container, optionally given *ref*."""
        monkeypatch.setattr(
            "womblex.store.build_info._CONTAINER_MARKERS", (Path("/"),),
        )
        if ref is None:
            monkeypatch.delenv(IMAGE_REF_ENV, raising=False)
        else:
            monkeypatch.setenv(IMAGE_REF_ENV, ref)

    def test_a_published_image_is_named_by_its_digest(self, monkeypatch):
        pinned = "ghcr.io/deepcivic/womblex@sha256:" + "a" * 64
        self._in_container(monkeypatch, pinned)
        info = image_info()
        assert info.containerised is True
        assert info.digest == "sha256:" + "a" * 64
        assert info.reference == pinned
        assert info.reason == ""

    def test_a_run_outside_a_container_says_so_rather_than_going_empty(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            "womblex.store.build_info._CONTAINER_MARKERS",
            (Path("/definitely-not-here"),),
        )
        monkeypatch.delenv(IMAGE_REF_ENV, raising=False)
        record = image_info().as_record()
        assert record["containerised"] is False
        assert record["digest"] is None
        assert record["reason"] == "not running in a container"

    def test_a_locally_built_image_records_the_absence_and_does_not_fail(
        self, monkeypatch,
    ):
        """What the local compose override produces: it builds, so it pins nothing."""
        self._in_container(monkeypatch, "")
        record = image_info().as_record()
        assert record["containerised"] is True
        assert record["digest"] is None
        assert IMAGE_REF_ENV in record["reason"]

    def test_a_tag_is_not_recorded_as_a_digest(self, monkeypatch):
        """A tag can be moved to point at other bytes, so it identifies nothing.

        Recording it in the digest field would be the one dishonest outcome
        available here — a moving pointer presented as an artefact's identity.
        """
        self._in_container(monkeypatch, "ghcr.io/deepcivic/womblex:0.5.12")
        info = image_info()
        assert info.digest == ""
        assert info.reference == "ghcr.io/deepcivic/womblex:0.5.12"
        assert "tag" in info.reason

    @pytest.mark.parametrize("reference", [
        "ghcr.io/x@",                       # separator, nothing after it
        "@sha256:" + "a" * 64,              # a digest naming no image
        "ghcr.io/x@sha256:abc",             # too short to be one
        "ghcr.io/x@SHA256:" + "A" * 64,     # digests are lower-case hex
        "a@b@c",                            # not a reference at all
    ])
    def test_a_malformed_reference_yields_no_digest(self, monkeypatch, reference):
        """A string that cannot be a digest is never recorded as one.

        The value is injected, so a deployment that sets it wrongly is the
        realistic failure. Recording whatever followed the separator would put
        a fabricated identifier in the record — the one thing this module is
        written not to do.
        """
        self._in_container(monkeypatch, reference)
        info = image_info()
        assert info.digest == ""
        assert "well-formed digest" in info.reason

    def test_the_record_carries_the_image_block(self, tmp_path, extraction, monkeypatch):
        pinned = "ghcr.io/deepcivic/womblex@sha256:" + "b" * 64
        self._in_container(monkeypatch, pinned)
        shards = _extracted(tmp_path, extraction)
        record = build_run_record(shards, read_manifest(shards))
        assert record["image"]["digest"] == "sha256:" + "b" * 64
        assert record["record_version"] == RUN_RECORD_VERSION

    def test_a_recorded_digest_still_declares_it_was_not_self_verified(
        self, tmp_path, extraction, monkeypatch,
    ):
        """The limitation is stated loudest when the digest IS present.

        The value is operator-supplied, so a record that named it without this
        would read as though the run had verified the image it was running in.
        """
        self._in_container(monkeypatch, "ghcr.io/x@sha256:" + "c" * 64)
        shards = _extracted(tmp_path, extraction)
        record = build_run_record(shards, read_manifest(shards))
        assert any("operator-honest" in gap for gap in record["partial"])

    def test_an_absent_digest_names_its_reason_in_partial(
        self, tmp_path, extraction, monkeypatch,
    ):
        self._in_container(monkeypatch, None)
        shards = _extracted(tmp_path, extraction)
        record = build_run_record(shards, read_manifest(shards))
        assert any(
            gap.startswith("image digest:") and IMAGE_REF_ENV in gap
            for gap in record["partial"]
        )


class TestModelCheck:
    """What the pre-run check said, observed from the files like stages and models."""

    @staticmethod
    def _remember(variant="paddleocr-v5", level="load"):
        from womblex.utils import model_check as mc

        mc._STATE[("ocr", "paddleocr")] = SlotCheck(
            "ocr", "paddleocr", ("extract",), CheckLevel(level), ok=True,
            source="builtin", variant=variant,
        )

    @classmethod
    def _checked(cls, tmp_path, extraction, **kw):
        cls._remember(**kw)
        return _extracted(tmp_path, extraction)

    def test_the_checks_come_from_the_files_footers(self, tmp_path, extraction):
        shards = self._checked(tmp_path, extraction)
        record = read_run_record(write_run_manifest(shards))
        (entry,) = record["model_check"]
        assert (entry["slot"], entry["name"], entry["variant"], entry["status"]) == (
            "ocr", "paddleocr", "paddleocr-v5", "ok",
        )
        assert entry["stages"] == ["extract"]
        assert not any("model check" in gap for gap in record["partial"])

    def test_two_variants_across_files_stay_two_entries(self, tmp_path, extraction):
        shards = self._checked(tmp_path, extraction, variant="paddleocr-v5")
        self._remember(variant="rapidocr-bundled-v4")
        _sidecar(shards, ".chunks.parquet", pa.table({"x": [1]}), "chunk")
        record = read_run_record(write_run_manifest(shards))
        assert {(e["variant"], tuple(e["stages"])) for e in record["model_check"]} == {
            ("paddleocr-v5", ("extract",)), ("rapidocr-bundled-v4", ("chunk",)),
        }

    def test_a_run_with_no_check_says_so_in_partial(self, tmp_path, extraction):
        record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
        assert record["model_check"] == []
        assert any("model check: no file records one" in g for g in record["partial"])

    def test_a_check_switched_off_is_named_as_such(self, tmp_path, extraction):
        check_models(_config(tmp_path), "off")
        record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
        assert record["model_check"] == []
        assert any("switched off" in g for g in record["partial"])

    def test_a_failed_check_keeps_its_reason(self, tmp_path, extraction):
        from womblex.utils import model_check as mc

        mc._STATE[("tokenizer", "x")] = SlotCheck(
            "tokenizer", "x", ("chunk",), CheckLevel.LOAD, ok=False, reason="not found",
        )
        record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
        (entry,) = record["model_check"]
        assert (entry["status"], entry["reason"]) == ("failed", "not found")

    def test_footers_a_caller_supplies_carry_the_check_too(self, tmp_path, extraction):
        """`womblex finalize` reads footers where the files live and hands them in."""
        shards = self._checked(tmp_path, extraction)
        table = read_manifest(shards / "batch-0001.parquet")
        record = build_run_record(shards, table, footers=_footers(shards))
        assert [e["slot"] for e in record["model_check"]] == ["ocr"]
