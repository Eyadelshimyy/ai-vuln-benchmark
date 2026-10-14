"""The vendored artifact worker-engine assets ship and are well-formed.

These are inlined into a large-detail artifact so it decodes + filters its
embedded Parquet offline (docs/architecture/large-detail-artifacts.md, increment 3). Like the
ECharts bundle, they live in ``tracebi/reports/assets`` and are force-included in
the wheel; this test guards their presence and shape so a build that drops them
fails in CI rather than shipping an artifact whose engine cannot load.
"""

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "tracebi" / "reports" / "assets"


def test_engine_worker_bundle_present_and_shaped():
    worker = ASSETS / "tracebi-engine.worker.js"
    assert worker.is_file(), "the worker engine bundle is missing"
    text = worker.read_text(encoding="utf-8", errors="replace")
    assert len(text) > 100_000, "worker bundle is implausibly small"
    # esbuild preserves the protocol's string literals; a bundle missing them
    # is not the engine worker.
    for marker in ("inited", "loaded", '"rows"', '"query"'):
        assert marker in text, f"worker bundle missing protocol marker {marker!r}"


def test_parquet_wasm_is_gzipped():
    wasm_gz = ASSETS / "parquet_wasm_bg.wasm.gz"
    assert wasm_gz.is_file(), "the gzipped parquet-wasm module is missing"
    data = wasm_gz.read_bytes()
    assert data[:2] == b"\x1f\x8b", "not a gzip stream"


def test_engine_notice_attributes_bundled_libraries():
    notice = ASSETS / "ENGINE_NOTICE.md"
    assert notice.is_file(), "ENGINE_NOTICE.md (attribution) is missing"
    text = notice.read_text(encoding="utf-8")
    for lib in ("parquet-wasm", "arquero", "apache-arrow"):
        assert lib in text, f"NOTICE does not attribute {lib}"


# ── inlining the engine into an artifact ─────────────────────────────────────

def test_engine_blocks_carry_worker_and_wasm_in_the_ids_the_runtime_reads():
    from tracebi.reports.embed import (
        ENGINE_WASM_ID, ENGINE_WORKER_ID, engine_blocks_html,
    )

    html = engine_blocks_html()
    assert f'id="{ENGINE_WORKER_ID}"' in html
    assert f'id="{ENGINE_WASM_ID}"' in html
    # The worker must be inert markup (the runtime reads textContent and builds
    # a blob) and must not be able to close its own block.
    assert 'type="text/plain"' in html
    body = html.split(f'id="{ENGINE_WORKER_ID}">', 1)[1].split("</script>", 1)[0]
    assert "</script" not in body


def test_engine_ships_only_when_a_binding_is_parquet():
    """A CSV artifact must not pay megabytes for an engine it never starts."""
    from tracebi.reports.stack import stack_tail

    csv_block = '<script id="tracebi-data-x" type="application/json">' \
                '{"name": "x", "columns": "[]", "dtypes": "[]", "csv": ""}</script>'
    pq_block = '<script id="tracebi-data-x" type="application/json">' \
               '{"name": "x", "format": "parquet", "parquet_b64": "AAA="}</script>'

    csv_tail = stack_tail(libs=(), data_blocks_html=csv_block)
    pq_tail = stack_tail(libs=(), data_blocks_html=pq_block)

    # Assert on the engine BLOCKS, not the bare ids: the runtime source itself
    # names both ids (it looks them up), so a substring check always matches.
    worker_block = 'id="tracebi-engine-worker"'
    wasm_block = 'id="tracebi-engine-wasm"'
    assert worker_block not in csv_tail
    assert worker_block in pq_tail
    assert wasm_block in pq_tail
    # The engine is megabytes; a CSV artifact must not carry it.
    assert len(pq_tail) - len(csv_tail) > 1_000_000


def test_csp_grants_wasm_and_blob_worker_but_never_js_eval():
    from tracebi.reports.embed import CSP

    # The worker engine needs a blob worker + WebAssembly compilation...
    assert "worker-src blob:" in CSP
    assert "'wasm-unsafe-eval'" in CSP
    assert "blob:" in CSP.split("script-src", 1)[1].split(";", 1)[0]
    # ...but JS eval is still never granted, and nothing may phone home.
    assert "'unsafe-eval'" not in CSP.replace("'wasm-unsafe-eval'", "")
    assert "connect-src 'none'" in CSP
