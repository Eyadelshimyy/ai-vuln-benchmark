"""``*.layout_regions.parquet``: schema, round trip, fingerprint and config."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.config import DatasetConfig, LayoutConfig, PathsConfig, WomblexConfig
from womblex.ingest.elements import BBox
from womblex.ingest.layout_step import LayoutRegion, PageLayout
from womblex.store.contract import ROLE_SENSITIVITY, read_footer_contract
from womblex.store.layout_output import (
    LAYOUT_REGIONS_SCHEMA,
    LayoutFingerprint,
    layout_fingerprint,
    layout_regions_path_for,
    layout_rows,
    read_footer_layout_fingerprint,
    read_footer_redaction_consumed,
    read_layout_regions,
    read_page_layouts,
    unfiltered_redaction_pages,
    write_layout_regions,
)


def _config(**layout: object) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
        layout=LayoutConfig(**layout),  # type: ignore[arg-type]
    )


def _fp(**over: object) -> LayoutFingerprint:
    base = {"model": "m", "model_digest": "d", "options_digest": "o", "dpi": 200,
            "page_scope": "consumers"}
    return LayoutFingerprint(**{**base, **over})  # type: ignore[arg-type]


class TestConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert (cfg.layout.model, cfg.layout.options, cfg.layout.page_scope) == (
            "pp-doclayout-m", {}, "consumers",
        )

    def test_page_scope_is_closed(self) -> None:
        with pytest.raises(ValueError):
            _config(page_scope="some")


class TestFingerprint:
    def test_equal_for_equal_config_and_moves_with_each_setting(self) -> None:
        base = layout_fingerprint(_config())
        assert base == layout_fingerprint(_config())
        assert base.model == "pp-doclayout-m" and base.dpi == 200
        assert base.model_digest.startswith("sha256:")
        assert layout_fingerprint(_config(page_scope="all")) != base
        assert layout_fingerprint(_config(options={"model_dir": "x"})).options_digest != (
            base.options_digest
        )

    def test_consumers_move_it_under_consumers_scope_only(self) -> None:
        cfg = _config()
        assert layout_fingerprint(cfg).consumers == "ocr,redaction"
        cfg.redaction.enabled = False
        assert layout_fingerprint(cfg).consumers == "ocr"
        assert layout_fingerprint(_config(page_scope="all")).consumers == ""
        legacy = {k: v for k, v in json.loads(_fp().to_json()).items() if k != "consumers"}
        assert LayoutFingerprint.from_json(json.dumps(legacy)) == _fp()  # a pre-consumers footer

    def test_dpi_is_the_ocr_dpi(self) -> None:
        cfg = _config()
        cfg.extraction.ocr.dpi = 300
        assert layout_fingerprint(cfg).dpi == 300

    def test_options_key_order_is_irrelevant(self) -> None:
        a = layout_fingerprint(_config(options={"a": 1, "b": 2}))
        b = layout_fingerprint(_config(options={"b": 2, "a": 1}))
        assert a.options_digest == b.options_digest

    def test_unknown_model_raises_listing_known_names(self) -> None:
        with pytest.raises(ValueError, match="pp-doclayout-m"):
            layout_fingerprint(_config(model="nope"))

    def test_footer_round_trip_and_garbage(self) -> None:
        fp = _fp()
        assert read_footer_layout_fingerprint(fp.footer_metadata()) == fp
        assert read_footer_layout_fingerprint({}) is None
        assert read_footer_layout_fingerprint(None) is None
        key = next(iter(fp.footer_metadata()))
        assert read_footer_layout_fingerprint({key: b"{not json"}) is None
        assert read_footer_layout_fingerprint({key: b'{"model": "m"}'}) is None


class TestSidecar:
    def _pages(self) -> list[tuple[str, list[PageLayout]]]:
        region = LayoutRegion(BBox(0.1, 0.2, 0.5, 0.25), "table", "table", 0.9)
        return [("h1", [
            PageLayout(0, "ok", [region, LayoutRegion(BBox(0, 0, 1, 0.1), "text", "paragraph", 0.5)]),
            PageLayout(1, "empty"),
            PageLayout(2, "error", error="RuntimeError: boom"),
        ])]

    def test_round_trip_keeps_status_rows_apart(self, tmp_path: Path) -> None:
        base = tmp_path / "batch-0001.parquet"
        write_layout_regions(layout_rows(self._pages()), base, _fp(), redaction_consumed=True)
        table = read_layout_regions(base)
        assert table.schema.equals(LAYOUT_REGIONS_SCHEMA)
        rows = table.to_pylist()
        assert [(r["page"], r["status"], r["region_order"]) for r in rows] == [
            (0, "ok", 0), (0, "ok", 1), (1, "empty", None), (2, "error", None),
        ]
        assert rows[0]["bbox"] == pytest.approx({"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.25})
        assert rows[0]["label"] == "table" and rows[0]["confidence"] == pytest.approx(0.9)
        assert rows[2]["bbox"] is None and rows[2]["confidence"] is None
        assert rows[3]["error"] == "RuntimeError: boom"

    def test_footer_carries_fingerprint_redaction_flag_and_contract(self, tmp_path: Path) -> None:
        base = tmp_path / "batch-0001.parquet"
        target = write_layout_regions([], base, _fp(), redaction_consumed=False)
        assert target == layout_regions_path_for(base)
        meta = pq.read_schema(str(target)).metadata
        assert read_footer_layout_fingerprint(meta) == _fp()
        assert read_footer_redaction_consumed(meta) is False
        assert read_footer_contract(meta)["sensitivity"] == "none"
        assert read_footer_redaction_consumed({}) is None

    def test_directory_read_concatenates_and_an_empty_one_is_typed(self, tmp_path: Path) -> None:
        assert read_layout_regions(tmp_path).num_rows == 0
        for n in (1, 2):
            write_layout_regions(
                layout_rows(self._pages()), tmp_path / f"batch-000{n}.parquet", _fp(),
                redaction_consumed=True,
            )
        assert read_layout_regions(tmp_path).num_rows == 8

    def test_a_shard_missing_a_column_is_refused(self, tmp_path: Path) -> None:
        import pyarrow as pa

        bad = tmp_path / "batch-0001.layout_regions.parquet"
        pq.write_table(pa.table({"source_hash": ["h"]}), bad)
        with pytest.raises(ValueError, match="missing columns"):
            read_layout_regions(bad)

    def test_contract_classifies_it_without_text(self) -> None:
        assert ROLE_SENSITIVITY["layout_regions"] == "none"


class TestReaders:
    def _write(self, tmp_path: Path, name: str, *, consumed: bool) -> None:
        pages = [("h1", [
            PageLayout(0, "ok", [LayoutRegion(BBox(0.1, 0.2, 0.5, 0.25), "table", "table", 0.9)]),
            PageLayout(1, "empty"),
            PageLayout(2, "error", error="RuntimeError: boom"),
        ])]
        write_layout_regions(
            layout_rows(pages), tmp_path / name, _fp(), redaction_consumed=consumed,
        )

    def test_page_layouts_invert_layout_rows(self, tmp_path: Path) -> None:
        self._write(tmp_path, "batch-0001.parquet", consumed=True)
        got = read_page_layouts(tmp_path)["h1"]
        assert {p: (v.status, len(v.regions)) for p, v in got.items()} == {
            0: ("ok", 1), 1: ("empty", 0), 2: ("error", 0),
        }
        b = got[0].regions[0].bbox
        assert (b.x, b.y, b.width, b.height) == pytest.approx((0.1, 0.2, 0.5, 0.25))
        assert got[0].regions[0].block_type == "table"
        assert got[2].error == "RuntimeError: boom"

    def test_unfiltered_pages_need_an_errored_page_and_a_consuming_footer(self, tmp_path: Path) -> None:
        self._write(tmp_path, "batch-0001.parquet", consumed=True)
        self._write(tmp_path, "batch-0002.parquet", consumed=False)
        assert unfiltered_redaction_pages(tmp_path) == [("h1", 2)]
        assert unfiltered_redaction_pages(tmp_path / "batch-0002.layout_regions.parquet") == []
