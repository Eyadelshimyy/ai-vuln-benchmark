"""The batch's layout step: scope selection, one call per page, the sidecar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

import womblex.utils.model_registry as reg
from tests._pdf_builders import PdfBuilder
from womblex.batch import process_batch
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.ingest.interfaces.protocols import LayoutRegionResult
from womblex.ingest.layout_step import LayoutSettings, run_layout_step, select_pages
from womblex.ingest.page_profile import profile_pages
from womblex.ingest.pdf.types import Rect
from womblex.store.layout_output import (
    read_footer_layout_fingerprint,
    read_footer_redaction_consumed,
    read_layout_regions,
)

_calls: list[tuple[int, int]] = []


class _Analyzer:
    """Two regions on every page, in a unit square of the render."""

    def __init__(self, fail_on_call: int | None = None, empty: bool = False) -> None:
        self.fail_on_call, self.empty = fail_on_call, empty

    def analyze(self, img: np.ndarray, conf_threshold: float = 0.3) -> list[LayoutRegionResult]:
        h, w = img.shape[:2]
        _calls.append((h, w))
        if self.fail_on_call == len(_calls):
            raise RuntimeError("boom")
        if self.empty:
            return []
        return [
            LayoutRegionResult((0.0, 0.0, w / 2, h / 4), "text", "paragraph", 0.9),
            LayoutRegionResult((w / 4, h / 2, w * 2, h), "table", "table", 0.8),  # runs off-page
        ]


@pytest.fixture(autouse=True)
def _fake_models():
    _calls.clear()
    reg.register(reg.SLOT_LAYOUT, "step-fake", lambda **_: _Analyzer(), source="test-dist")
    reg.register(reg.SLOT_LAYOUT, "step-empty", lambda **_: _Analyzer(empty=True), source="test-dist")


def _config(tmp_path: Path, *, model: str = "step-fake", scope: str = "consumers",
            redaction: bool = False, filter_on: bool = True) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
        layout={"model": model, "page_scope": scope},
        redaction={"enabled": redaction, "use_layout_filter": filter_on},
    )


_PROSE = "A native text layer, long enough that the page profiler routes it as native. " * 4


def _mixed_pdf(path: Path) -> Path:
    """Page 0 native text; page 1 an image only (OCR-routed); page 2 a vector redaction."""
    img = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
    return (
        PdfBuilder(path)
        .page().text(72, 100, _PROSE)
        .page().image(Rect(50, 50, 450, 450), img)
        .page().text(72, 100, _PROSE).rect(Rect(72, 200, 300, 230))
        .save()
    )


def _select(tmp_path: Path, **kw) -> list[int]:
    cfg = _config(tmp_path, **kw)
    with _open(_mixed_pdf(tmp_path / "m.pdf")) as doc:
        return select_pages(doc, profile_pages(doc), LayoutSettings.from_config(cfg), "paddleocr")


def _open(path: Path):
    from womblex.ingest.pdf import open_document

    return open_document(path)


class TestSelection:
    def test_consumers_without_redaction_is_the_ocr_pages(self, tmp_path: Path) -> None:
        assert _select(tmp_path) == [1]

    def test_the_redaction_filter_adds_pages_without_vector_redactions(self, tmp_path: Path) -> None:
        assert _select(tmp_path, redaction=True) == [0, 1]

    def test_the_filter_off_leaves_redaction_out_of_it(self, tmp_path: Path) -> None:
        assert _select(tmp_path, redaction=True, filter_on=False) == [1]

    def test_all_is_every_page(self, tmp_path: Path) -> None:
        assert _select(tmp_path, scope="all") == [0, 1, 2]

    def test_unreadable_drawings_select_the_page_rather_than_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_a, **_k):
            raise RuntimeError("bad content stream")

        monkeypatch.setattr("womblex.redact.stage._detect_vector_redactions", _boom)
        assert _select(tmp_path, redaction=True) == [0, 1, 2]

    def test_a_markdown_engine_bypasses_layout_for_ocr_pages(self, tmp_path: Path) -> None:
        reg.register(reg.SLOT_OCR, "step-md", lambda **_: object(), source="test-dist",
                     traits={"markdown": True})
        cfg = _config(tmp_path)
        with _open(_mixed_pdf(tmp_path / "m.pdf")) as doc:
            assert select_pages(doc, profile_pages(doc), LayoutSettings.from_config(cfg), "step-md") == []


class TestRun:
    def _run(self, tmp_path: Path, **kw):
        cfg = _config(tmp_path, **kw)
        with _open(_mixed_pdf(tmp_path / "m.pdf")) as doc:
            return run_layout_step(doc, profile_pages(doc), LayoutSettings.from_config(cfg), "paddleocr")

    def test_one_analyser_call_per_selected_page(self, tmp_path: Path) -> None:
        out = self._run(tmp_path, scope="all")
        assert [p.page for p in out.pages] == [0, 1, 2] and len(_calls) == 3

    def test_boxes_are_normalised_and_clipped_to_the_page(self, tmp_path: Path) -> None:
        page = self._run(tmp_path).pages[0]
        assert page.status == "ok"
        a, b = (r.bbox for r in page.regions)
        assert (a.x, a.y, a.width, a.height) == pytest.approx((0, 0, 0.5, 0.25))
        assert (b.x, b.y, b.width, b.height) == pytest.approx((0.25, 0.5, 0.75, 0.5))

    def test_no_regions_is_empty_not_error(self, tmp_path: Path) -> None:
        page = self._run(tmp_path, model="step-empty").pages[0]
        assert (page.status, page.regions, page.error) == ("empty", [], "")

    def test_a_failing_page_is_an_error_row_and_the_rest_continue(self, tmp_path: Path) -> None:
        reg.register(reg.SLOT_LAYOUT, "step-flaky2", lambda **_: _Analyzer(fail_on_call=2),
                     source="test-dist")
        out = self._run(tmp_path, model="step-flaky2", scope="all")
        by = {p.page: p for p in out.pages}
        assert by[1].status == "error" and "boom" in by[1].error  # the second call
        assert by[0].status == "ok" and by[2].status == "ok"

    def test_a_model_that_cannot_build_errors_every_selected_page(self, tmp_path: Path) -> None:
        def broken(**_):
            raise FileNotFoundError("weights gone")

        reg.register(reg.SLOT_LAYOUT, "step-broken", broken, source="test-dist")
        out = self._run(tmp_path, model="step-broken", scope="all")
        assert {p.status for p in out.pages} == {"error"}
        assert all("weights gone" in p.error for p in out.pages)


class TestBatch:
    def test_sidecar_and_elements_footer_carry_the_same_fingerprint(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, redaction=True)
        src = _mixed_pdf(tmp_path / "m.pdf")
        csv = tmp_path / "t.csv"
        csv.write_text("a,b\n1,2\n")
        shards = tmp_path / "shards"
        process_batch([src, csv], cfg, batch_num=1, shard_dir=shards)

        sidecar = shards / "batch-0001.layout_regions.parquet"
        elements = shards / "batch-0001.elements.parquet"
        fp = read_footer_layout_fingerprint(pq.read_schema(str(sidecar)).metadata)
        assert fp is not None and fp.model == "step-fake" and fp.dpi == 200
        assert read_footer_layout_fingerprint(pq.read_schema(str(elements)).metadata) == fp
        assert read_footer_redaction_consumed(pq.read_schema(str(sidecar)).metadata) is True
        # Only the sidecar and the elements say what layout was consumed.
        for role in ("table_cells", "form_fields", "_manifest"):
            meta = pq.read_schema(str(shards / f"batch-0001.{role}.parquet")).metadata
            assert read_footer_layout_fingerprint(meta) is None

        table = read_layout_regions(sidecar)
        assert set(table["source_hash"].to_pylist()) and {r for r in table["page"].to_pylist()} == {0, 1}
        # The CSV contributes no rows.
        manifest = pq.read_table(str(shards / "batch-0001._manifest.parquet")).to_pylist()
        csv_hash = next(r["source_hash"] for r in manifest if r["filename"] == "t.csv")
        assert csv_hash not in set(table["source_hash"].to_pylist())

    def test_elements_do_not_change_with_the_layout_step(self, tmp_path: Path) -> None:
        from womblex.store.content_digest import content_digest

        src = _mixed_pdf(tmp_path / "m.pdf")
        digests = []
        for scope in ("consumers", "all"):
            cfg = _config(tmp_path, scope=scope)
            out = process_batch([src], cfg, batch_num=1, shard_dir=tmp_path / scope)
            digests.append(content_digest(out.batch.results[0].extraction.elements))
        assert digests[0] == digests[1]
