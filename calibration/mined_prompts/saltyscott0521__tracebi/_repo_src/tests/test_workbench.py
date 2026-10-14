"""
The workbench — the artifact dev loop's shared state (architecture v2 §2.5).

show() is the notebook-cell-output primitive: a NO-OP outside dev (no
TRACEBI_WORKBENCH_DIR → builds and CI ignore it) and never raising inside
it. collect_state is THE one state builder every workbench surface renders
from — per-binding errors are captured into the state, never raised, and
everything here is dev-state only: no receipts, nothing in builds.
"""

import json
import os
import sys

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.workbench import (
    ACTIVE_FILE,
    EXHIBIT_CAP,
    collect_discovery_state,
    collect_state,
    discovery_dir,
    frame_profile,
    heartbeat,
    last_seq,
    read_exhibits,
    read_pins,
    show,
    workbench_dir,
    write_pins,
)


@pytest.fixture()
def wb_model():
    df = pd.DataFrame({"region": ["NE", "SE"], "revenue": [100.0, 250.0]})
    m = DataModel("wb_model")
    m.add_connector(MemoryConnector("wb_mem", tables={"t": df}))
    m.add_table("t", connector="wb_mem", source="t")
    m.add_dimension("dim_r", table_name="t", key_col="region",
                    attributes=["region"])
    m.add_fact("f", table_name="t", measures=["revenue"], foreign_keys={})
    m.add_measure("total", column="revenue", agg="sum")
    m.connect()
    return m


def _pkg(parent, name="wbpkg", broken_binding=True):
    """A working-state package: a spare binding no figure uses, a binding
    naming a model that does not exist, an unverified figure, a report.py
    output, and two prose numbers outside any figure."""
    pkg = parent / name
    pkg.mkdir(parents=True, exist_ok=True)
    data = {
        "kpi": {"model": "wb_model",
                "query": {"fact": "f", "measures": ["total"]}},
        "by_region": {"model": "wb_model",
                      "query": {"fact": "f", "measures": {"revenue": "sum"},
                                "dimensions": ["dim_r.region"]}},
        "spare": {"model": "wb_model",
                  "query": {"fact": "f", "measures": ["total"]}},
    }
    figures = (
        '<div data-tb-figure="value" data-tb-binding="kpi" '
        'data-tb-cell="total" id="fig-kpi"></div>'
        '<table data-tb-figure="table" data-tb-binding="by_region" '
        'id="fig-tbl"></table>'
        '<table data-tb-figure="table" data-tb-binding="ranked" '
        'id="fig-rank"></table>'
        '<div data-tb-figure="value" data-tb-unverified '
        'data-tb-note="estimate" id="fig-est">42</div>'
    )
    if broken_binding:
        data["broken"] = {"model": "ghost_model",
                          "query": {"fact": "f", "measures": ["total"]}}
        figures += ('<div data-tb-figure="chart" data-tb-binding="broken" '
                    'id="fig-broken"></div>')
    (pkg / "report.json").write_text(json.dumps({"name": name, "data": data}))
    (pkg / "template.html").write_text(
        "<html><head><title>x</title></head><body>"
        "<p>Revenue rose 12.5 over 2024.</p>" + figures + "</body></html>"
    )
    (pkg / "report.py").write_text(
        "def build(frames):\n"
        "    return {'ranked': frames['by_region'].head(1)}\n")
    return pkg


class TestShow:
    def test_noop_without_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRACEBI_WORKBENCH_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        show(pd.DataFrame({"a": [1]}), note="ignored")
        show("also ignored")
        assert list(tmp_path.iterdir()) == []   # a build/CI run leaves no trace

    def test_writes_frame_and_note_jsonl(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        df = pd.DataFrame({"fund": ["a", "b"], "fv": [1.5, float("nan")]})
        show(df, note="after dropping nulls", name="marks")
        show("## working note")

        lines = [json.loads(line) for line in
                 open(os.path.join(wb, "exhibits.jsonl"), encoding="utf-8")]
        assert [e["seq"] for e in lines] == [1, 2]     # monotonic
        frame, note = lines
        assert frame["kind"] == "frame"
        assert frame["name"] == "marks" and frame["note"] == "after dropping nulls"
        assert frame["columns"] == ["fund", "fv"] and frame["shape"] == [2, 2]
        assert frame["rows"][1]["fv"] is None          # NaN is JSON-safe null
        # Under pytest sys.argv[0] is a real file, so `source` (the
        # producing script — TestExhibitSource) is present; set it aside.
        note = {k: v for k, v in note.items()
                if k not in ("source", "origin", "step")}
        assert note == {"seq": 2, "at": note["at"], "kind": "note",
                        "text": "## working note"}
        # Newest first, and the pin anchor tracks the newest seq.
        assert [e["seq"] for e in read_exhibits(wb)] == [2, 1]
        assert last_seq(wb) == 2

    def test_show_never_raises(self, tmp_path, monkeypatch, capsys):
        blocker = tmp_path / "blocker"
        blocker.write_text("")                          # a FILE where a dir must go
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", str(blocker / "sub"))
        show(pd.DataFrame({"a": [1]}))                  # must not raise
        assert "exhibit dropped" in capsys.readouterr().err


class TestShowDiscoveryHeartbeat:
    """The posting rule, exactly: TRACEBI_WORKBENCH_DIR always wins when
    set. Without it, show() posts to the CURRENT working directory's
    _discovery workbench only while its .active heartbeat is fresh — no
    live server (builds, CI) means stale/absent, and show() stays a no-op."""

    def test_fresh_heartbeat_gates_show_into_discovery(
            self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRACEBI_WORKBENCH_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        wb = discovery_dir(str(tmp_path))
        heartbeat(wb)                       # a discovery server is "live"
        show(pd.DataFrame({"a": [1]}), note="probe output")
        exhibits = read_exhibits(wb)
        assert [e["kind"] for e in exhibits] == ["frame"]
        assert exhibits[0]["note"] == "probe output"

    def test_stale_heartbeat_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRACEBI_WORKBENCH_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        wb = discovery_dir(str(tmp_path))
        heartbeat(wb)
        marker = os.path.join(wb, ACTIVE_FILE)
        past = os.path.getmtime(marker) - 60        # server long gone
        os.utime(marker, (past, past))
        show(pd.DataFrame({"a": [1]}))
        assert read_exhibits(wb) == []      # the build/CI no-op guarantee

    def test_env_var_wins_over_the_heartbeat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wb = discovery_dir(str(tmp_path))
        heartbeat(wb)
        explicit = str(tmp_path / "explicit_wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", explicit)
        show("a session note")
        assert [e["kind"] for e in read_exhibits(explicit)] == ["note"]
        assert read_exhibits(wb) == []


class TestChartExhibits:
    """show(chart=...) — the sketch stays a workbench exhibit: the frame's
    excerpt plus a recipe the page renders live. Validation is soft by the
    never-raise contract: a bad sketch degrades to a plain frame exhibit."""

    def _df(self):
        return pd.DataFrame({"sector": ["fin", "tech", "fin"],
                             "fair_value": [10.0, 20.0, 30.0]})

    def test_chart_posts_recipe_and_profile(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show(self._df(), chart="bar", x="sector", y="fair_value",
             note="fv by sector")
        [entry] = read_exhibits(wb)
        assert entry["kind"] == "chart"
        assert entry["recipe"] == {"chart": "bar", "x": "sector",
                                   "y": ["fair_value"]}
        assert entry["note"] == "fv by sector"
        # The excerpt travels with the recipe — same shape as a frame.
        assert entry["columns"] == ["sector", "fair_value"]
        assert entry["shape"] == [3, 2]
        assert entry["rows"][0] == {"sector": "fin", "fair_value": 10.0}
        # And the profile is computed over the full frame.
        assert entry["profile"]["fair_value"]["mean"] == 20.0
        assert entry["profile"]["sector"]["top"][0] == "fin"

    def test_unknown_chart_kind_falls_back_to_frame(
            self, tmp_path, monkeypatch, capsys):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show(self._df(), chart="sunburst", x="sector", y="fair_value")
        [entry] = read_exhibits(wb)
        assert entry["kind"] == "frame"
        assert "recipe" not in entry
        assert "profile" in entry               # still a full frame exhibit
        assert "fell back to frame" in capsys.readouterr().err

    def test_missing_or_unknown_axes_fall_back_to_frame(
            self, tmp_path, monkeypatch, capsys):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show(self._df(), chart="bar")                       # no x/y at all
        show(self._df(), chart="bar", x="sector", y="ghost")  # not a column
        assert [e["kind"] for e in read_exhibits(wb)] == ["frame", "frame"]
        assert capsys.readouterr().err.count("fell back to frame") == 2


class TestFrameProfile:
    def test_mixed_dtype_profile(self):
        df = pd.DataFrame({
            "fund": ["alpha", "beta", "alpha", None],
            "fv": [1.0, 2.0, float("nan"), 6.0],
            "qty": [1, 2, 3, 4],
        })
        p = frame_profile(df)
        fund = p["fund"]
        # The profile DISPLAYS what pandas reports, so the string dtype's
        # name differs by pandas major ('object' on 2.x, 'str' on 3.x) —
        # unlike fingerprints, which canonicalize. Accept either label.
        assert fund.pop("dtype") in ("object", "str")
        assert fund == {"nulls": 1, "distinct": 2, "top": ["alpha", "beta"]}
        fv = p["fv"]
        assert fv["dtype"] == "float64"
        assert fv["nulls"] == 1 and fv["distinct"] == 3
        assert (fv["min"], fv["max"], fv["mean"]) == (1.0, 6.0, 3.0)
        qty = p["qty"]
        assert (qty["min"], qty["max"], qty["mean"]) == (1.0, 4.0, 2.5)
        json.dumps(p)                           # JSON-safe throughout

    def test_all_nan_numeric_is_nan_safe(self):
        p = frame_profile(pd.DataFrame({"empty": [float("nan")] * 3}))
        assert p["empty"]["nulls"] == 3 and p["empty"]["distinct"] == 0
        assert p["empty"]["min"] is None
        assert p["empty"]["max"] is None
        assert p["empty"]["mean"] is None


class TestPins:
    def test_pins_round_trip_via_file(self, tmp_path):
        wb = str(tmp_path)
        pins = [{"id": "fig-kpi", "note": "keep this one", "at_seq": 3}]
        write_pins(wb, pins)
        assert read_pins(wb) == pins
        assert read_pins(str(tmp_path / "missing")) == []

class TestCollectState:
    @pytest.fixture()
    def state(self, wb_model, tmp_path, monkeypatch):
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", str(tmp_path / "wb"))
        return collect_state(str(_pkg(tmp_path)), {"wb_model": wb_model})

    def test_figures_carry_provenance_and_unbound_flags(self, state):
        figs = {f["id"]: f for f in state["figures"]}
        assert figs["fig-kpi"]["provenance"] == "verified"
        assert figs["fig-tbl"]["provenance"] == "verified"
        assert figs["fig-rank"]["provenance"] == "derived"   # report.py — never green
        assert figs["fig-est"]["provenance"] == "unverified"
        assert figs["fig-broken"].get("unbound") is True     # its binding failed
        assert all(f["pinned"] is False for f in state["figures"])

    def test_coverage_counts(self, state):
        assert state["coverage"] == {"total": 5, "verified": 2, "derived": 1,
                                     "unverified": 1, "unbound_errors": 1}

    def test_per_binding_error_is_captured_not_raised(self, state):
        broken = next(b for b in state["bindings"] if b["name"] == "broken")
        assert "ghost_model" in broken["error"]
        assert broken["used_by"] == ["fig-broken"]
        # The healthy bindings still resolved fully alongside it.
        kpi = next(b for b in state["bindings"] if b["name"] == "kpi")
        assert kpi["source"] == "query" and kpi["rows"] == 1
        assert len(kpi["fingerprint"]) == 12
        assert kpi["preview"] == [{"total": 350.0}]
        ranked = next(b for b in state["bindings"] if b["name"] == "ranked")
        assert ranked["source"] == "python" and ranked["used_by"] == ["fig-rank"]

    def test_unused_bindings_and_lint(self, state):
        assert state["unused_bindings"] == ["spare"]
        # "12.5" and "2024" sit in prose; the 42 inside fig-est is claimed.
        assert state["lint"]["numeric_literals_outside_figures"] == 2

    def test_a_totals_binding_is_not_unused(self, wb_model, tmp_path, monkeypatch):
        # A binding read only by a table's totals row (data-tb-totals) is used.
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", str(tmp_path / "wb"))
        pkg = _pkg(tmp_path)
        tpl = (pkg / "template.html").read_text().replace(
            'data-tb-binding="by_region" ',
            'data-tb-binding="by_region" data-tb-totals="spare" ')
        (pkg / "template.html").write_text(tpl)
        state = collect_state(str(pkg), {"wb_model": wb_model})
        assert state["unused_bindings"] == []

    def test_pins_merge_into_figures(self, wb_model, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        write_pins(wb, [{"id": "fig-kpi", "note": "the headline", "at_seq": 0}])
        state = collect_state(str(_pkg(tmp_path)), {"wb_model": wb_model})
        figs = {f["id"]: f for f in state["figures"]}
        assert figs["fig-kpi"]["pinned"] is True
        assert figs["fig-tbl"]["pinned"] is False
        assert state["pins"][0]["note"] == "the headline"

_MODEL_SOURCE = """\
import pandas as pd
from tracebi import DataModel, MemoryConnector

df = pd.DataFrame({"region": ["NE", "SE"], "revenue": [100.0, %s]})
model = DataModel("wb_model")
model.add_connector(MemoryConnector("wb_mem", tables={"t": df}))
model.add_table("t", connector="wb_mem", source="t")
model.add_dimension("dim_r", table_name="t", key_col="region",
                    attributes=["region"])
model.add_fact("f", table_name="t", measures=["revenue"], foreign_keys={})
model.add_measure("total", column="revenue", agg="sum")
model.connect()
"""


class TestDevServerPackageRender:
    """The dev server's render function, in-process — the working state is
    served from memory (exploration kept, stage meta, data blocks), and the
    feed's auto-entries appear only when a binding's fingerprint CHANGED."""

    def _project(self, tmp_path, monkeypatch):
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "wb_model.py").write_text(_MODEL_SOURCE % "250.0")
        pkg = _pkg(tmp_path / "reports", broken_binding=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TRACEBI_WORKBENCH_DIR", raising=False)
        # As _serve() sets it: a .pyc written for the model file would read
        # "valid" against a same-second same-size edit and exec stale code.
        import sys
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        return pkg

    def test_render_serves_exploration_page_from_memory(
            self, tmp_path, monkeypatch):
        from tracebi._dev_server import _PackageTarget
        pkg = self._project(tmp_path, monkeypatch)
        page = _PackageTarget(pkg).render()
        assert 'content="exploration"' in page          # stage meta — never final
        assert "tracebi-data-kpi" in page               # stamped bytes embedded
        assert "tracebi-data-ranked" in page            # report.py output beside them
        # Serving leaked no env var and wrote no build output.
        assert "TRACEBI_WORKBENCH_DIR" not in os.environ
        assert not (tmp_path / "output").exists()

    def test_broken_package_renders_the_error_page(self, tmp_path, monkeypatch):
        from tracebi._dev_server import _PackageTarget
        pkg = self._project(tmp_path, monkeypatch)
        (pkg / "report.py").write_text("def build(frames):\n    return {}\n")
        page = _PackageTarget(pkg).render()
        assert "Report package failed" in page
        assert "non-empty dict" in page                 # the real traceback, escaped

    def test_auto_entry_only_when_fingerprint_changed(
            self, tmp_path, monkeypatch):
        from tracebi._dev_server import _PackageTarget
        pkg = self._project(tmp_path, monkeypatch)
        target = _PackageTarget(pkg)
        target.render()
        assert read_exhibits(target.wb_dir) == []       # first build only primes
        target.render()
        assert read_exhibits(target.wb_dir) == []       # unchanged data — silence
        (tmp_path / "models" / "wb_model.py").write_text(_MODEL_SOURCE % "999.0")
        target.render()
        texts = [e["text"] for e in read_exhibits(target.wb_dir)]
        assert texts and all(t.startswith("binding ") for t in texts)
        assert any("kpi updated" in t for t in texts)


_DISCOVERY_MODEL = """\
from tracebi import DataModel
from tracebi.connectors.duckdb_connector import DuckDBConnector

# Lazy by design: no connect() at import — discovery must list the model
# even before phase 1 has ever run.
model = DataModel("disc_model")
model.add_connector(DuckDBConnector("wh", database="data/warehouse.duckdb"))
model.add_table("fact_holdings", connector="wh", source="fact_holdings")
model.add_table("dim_issuer", connector="wh", source="dim_issuer")
model.add_relationship("h_to_i", left_table="fact_holdings",
                       right_table="dim_issuer", left_key="issuer_id")
model.add_dimension("dim_issuer", table_name="dim_issuer",
                    key_col="issuer_id", attributes=["issuer"])
model.add_fact("holdings", table_name="fact_holdings",
               measures=["fair_value"],
               foreign_keys={"dim_issuer": "issuer_id"})
model.add_measure("total_fv", column="fair_value", agg="sum")
"""


class TestCollectDiscoveryState:
    """The project-level state builder — phases ① and ② made visible: the
    warehouse (tables + sink-contract summaries), the models' declared star
    schemas, the packages, and the _discovery feed. Per-panel failures are
    captured into the state, never raised."""

    @pytest.fixture()
    def discovery_project(self, tmp_path):
        pytest.importorskip("duckdb")
        from tracebi.connectors.duckdb_connector import DuckDBConnector
        from tracebi.contracts import contract

        (tmp_path / "data").mkdir()
        wh = str(tmp_path / "data" / "warehouse.duckdb")
        sink = DuckDBConnector("wh", database=wh)
        sink.write(pd.DataFrame({"issuer_id": [1, 2, 3],
                                 "fair_value": [10.0, 20.0, 30.0]}),
                   "fact_holdings")
        sink.write(pd.DataFrame({"issuer_id": [1, 2, 3],
                                 "issuer": ["a", "b", "c"]}), "dim_issuer")
        with contract("holdings", warehouse=wh) as c:
            c.rows("fact_holdings", exactly=3)
            c.unique("dim_issuer", ["issuer_id"])

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "disc_model.py").write_text(_DISCOVERY_MODEL)
        (tmp_path / "models" / "broken_model.py").write_text(
            "import nope_missing\n")
        (tmp_path / "reports" / "empty_pkg").mkdir(parents=True)
        return tmp_path

    def test_collects_warehouse_models_packages(self, discovery_project):
        state = collect_discovery_state(str(discovery_project), {})
        assert state["mode"] == "discovery"
        assert state["name"] == "_discovery"

        wh = state["warehouse"]
        assert wh["exists"] is True
        assert wh["path"].endswith(os.path.join("data", "warehouse.duckdb"))
        tables = {t["name"]: t for t in wh["tables"]}
        assert tables["fact_holdings"]["rows"] == 3
        assert set(tables["fact_holdings"]["columns"]) == {"issuer_id",
                                                           "fair_value"}
        assert tables["dim_issuer"]["rows"] == 3

        # The sink-contract summary — the sink satisfied its contract.
        holdings = wh["contracts"]["holdings"]
        assert holdings["checks"] == "2/2"
        assert holdings["tables"] == ["dim_issuer", "fact_holdings"]
        assert holdings["checked_at"]

        # Model listing via the public info(); the broken FILE is an error
        # entry in the state, not an exception out of it.
        by_name = {m["name"]: m for m in state["models"]}
        good = by_name["disc_model"]
        assert [f["name"] for f in good["facts"]] == ["holdings"]
        assert [d["name"] for d in good["dimensions"]] == ["dim_issuer"]
        assert [m["name"] for m in good["measures"]] == ["total_fv"]
        broken = by_name["broken_model"]
        assert "nope_missing" in broken["error"]

        assert state["packages"] == ["empty_pkg"]
        assert state["exhibits"] == [] and state["pins"] == []

    def test_loaded_models_are_not_reloaded_and_dedupe(self, discovery_project):
        """A model the caller already loaded (keyed by stem AND .name, as
        the loaders do) appears once, and its declared warehouse path
        dedupes against data/warehouse.duckdb."""
        from tracebi.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.auto_discover(str(discovery_project / "models"))
        m = reg.get("disc_model")
        state = collect_discovery_state(
            str(discovery_project), {"disc_model": m, "alias": m})
        names = [e["name"] for e in state["models"]]
        assert names.count("disc_model") == 1
        assert state["warehouse"]["exists"] is True

    def test_warehouse_tables_carry_profiles(self, discovery_project):
        """The same profile shape as frame_profile, via read-only SQL
        aggregates — the frame itself never leaves the warehouse."""
        state = collect_discovery_state(str(discovery_project), {})
        tables = {t["name"]: t for t in state["warehouse"]["tables"]}

        fv = tables["fact_holdings"]["profile"]["fair_value"]
        assert fv["nulls"] == 0 and fv["distinct"] == 3
        assert (fv["min"], fv["max"], fv["mean"]) == (10.0, 30.0, 20.0)

        issuer = tables["dim_issuer"]["profile"]["issuer"]
        assert issuer["dtype"] == "VARCHAR"
        assert issuer["nulls"] == 0 and issuer["distinct"] == 3
        assert sorted(issuer["top"]) == ["a", "b", "c"]
        # Numeric keys profile too (BIGINT lands in the numeric branch).
        key = tables["dim_issuer"]["profile"]["issuer_id"]
        assert (key["min"], key["max"], key["mean"]) == (1.0, 3.0, 2.0)

    def test_wide_table_gets_profile_null(self, tmp_path):
        pytest.importorskip("duckdb")
        from tracebi.connectors.duckdb_connector import DuckDBConnector

        (tmp_path / "data").mkdir()
        wh = str(tmp_path / "data" / "warehouse.duckdb")
        wide = pd.DataFrame({f"c{i}": [1] for i in range(51)})
        DuckDBConnector("wh", database=wh).write(wide, "wide")
        state = collect_discovery_state(str(tmp_path), {})
        [t] = state["warehouse"]["tables"]
        assert t["rows"] == 1 and len(t["columns"]) == 51
        assert t["profile"] is None
        assert "51 columns" in t["note"]

    def test_missing_warehouse_is_state_not_crash(self, tmp_path):
        state = collect_discovery_state(str(tmp_path), {})
        wh = state["warehouse"]
        assert wh["exists"] is False
        assert wh["tables"] == [] and wh["contracts"] is None
        assert state["models"] == [] and state["packages"] == []

class TestNoteMarkdown:
    """Notebook-cell notes: markdown renders in the feed, escaped-first."""

    def test_note_exhibits_gain_preescaped_html(self):
        from tracebi.workbench import render_note_markdown
        out = render_note_markdown([
            {"kind": "note", "seq": 1,
             "text": "## Approach\n\nDropped **9** null-mark funds."},
            {"kind": "frame", "seq": 2, "rows": []},
        ])
        assert "<h3>Approach</h3>" in out[0]["html"]
        assert "<strong>9</strong>" in out[0]["html"]
        assert "html" not in out[1], "only notes convert"

    def test_markdown_cannot_smuggle_markup(self):
        from tracebi.workbench import render_note_markdown
        out = render_note_markdown([
            {"kind": "note", "seq": 1, "text": "<script>alert(1)</script>"},
        ])
        assert "<script>" not in out[0]["html"]
        assert "&lt;script&gt;" in out[0]["html"]


class TestExhibitSource:
    """show() records the producing script per exhibit — the notebook's
    'which cell': sys.argv[0], cwd-relative when derivable, absolute
    otherwise, omitted for interactive/no-file contexts."""

    def test_source_is_cwd_relative_when_derivable(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        script = tmp_path / "transforms" / "probe.py"
        script.parent.mkdir()
        script.write_text("# the probe")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", [str(script)])
        show("from the probe")
        [entry] = read_exhibits(wb)
        assert entry["source"] == os.path.join("transforms", "probe.py")

    def test_source_falls_back_to_absolute_outside_cwd(
            self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        script = tmp_path / "elsewhere" / "probe.py"
        script.parent.mkdir()
        script.write_text("# the probe")
        inside = tmp_path / "project"
        inside.mkdir()
        monkeypatch.chdir(inside)
        monkeypatch.setattr(sys, "argv", [str(script)])
        show("from outside the project")
        [entry] = read_exhibits(wb)
        assert entry["source"] == str(script)

    def test_source_omitted_for_interactive(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        monkeypatch.setattr(sys, "argv", [""])
        show("typed at a prompt")
        [entry] = read_exhibits(wb)
        assert "source" not in entry


def test_workbench_page_carries_a_csp():
    """The dev workbench does md.innerHTML of md_to_html output. Beyond the
    escaping, the page must carry a CSP as defense-in-depth — relaxed to
    connect-src 'self' for its /__workbench/state.json poll."""
    from tracebi._dev_server import _workbench_page

    page = _workbench_page("demo")
    assert 'http-equiv="Content-Security-Policy"' in page
    assert "default-src 'none'" in page
    assert "connect-src 'self'" in page      # the status poll needs it
    assert "__CSP__" not in page             # placeholder must be filled


class TestSessionExport:
    """export_session — the FULL feed as ONE exploration-record HTML: honest
    by construction (stage meta + banner, no manifest, verify refuses it),
    escaped throughout, uncapped by EXHIBIT_CAP."""

    def _session(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        monkeypatch.chdir(tmp_path)
        script = tmp_path / "transforms" / "probe.py"
        script.parent.mkdir(exist_ok=True)
        script.write_text("# probe")
        monkeypatch.setattr(sys, "argv", [str(script)])
        show("## Approach\n\nDropped **9** null-mark funds.")
        show(pd.DataFrame({"issuer": ["<script>alert(1)</script>", "b"],
                           "fv": [1.5, 2.5]}), note="hostile & clean")
        show(pd.DataFrame({"sector": ["fin", "tech"], "fv": [10.0, 20.0]}),
             chart="bar", x="sector", y="fv")
        write_pins(wb, [{"id": "fig-x", "note": "keep this cut", "at_seq": 2}])
        return wb

    def _export(self, wb, tmp_path, title=None):
        from tracebi._session_export import export_session
        out = tmp_path / "explorations" / "record.html"
        export_session(wb, str(out), title=title)
        return out, out.read_text(encoding="utf-8")

    def test_stage_meta_and_banner(self, tmp_path, monkeypatch):
        wb = self._session(tmp_path, monkeypatch)
        _out, html = self._export(wb, tmp_path)
        assert '<meta name="tracebi-stage" content="exploration">' in html
        assert "Exploration record — a lab notebook, not a report" in html
        assert "carry no receipts" in html

    def test_carries_a_strict_csp(self, tmp_path, monkeypatch):
        """The offline record interpolates md_to_html output; escaping is the
        real defense, but the file must also carry the strict CSP so a future
        escaping regression can't reach external script or exfil."""
        wb = self._session(tmp_path, monkeypatch)
        _out, html = self._export(wb, tmp_path)
        assert 'http-equiv="Content-Security-Policy"' in html
        assert "default-src 'none'" in html
        assert "connect-src 'none'" in html      # an offline file never polls

    def test_notes_render_markdown_and_frames_escape(
            self, tmp_path, monkeypatch):
        wb = self._session(tmp_path, monkeypatch)
        _out, html = self._export(wb, tmp_path)
        assert "<h3>Approach</h3>" in html               # markdown-converted
        assert "<strong>9</strong>" in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "<script>alert(1)" not in html            # escaped, not live

    def test_export_is_uncapped(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        monkeypatch.chdir(tmp_path)
        n = EXHIBIT_CAP + 20
        for i in range(n):
            show(f"note {i}")
        assert len(read_exhibits(wb)) == EXHIBIT_CAP     # the DISPLAY cap
        _out, html = self._export(wb, tmp_path)
        assert "#1</span>" in html                       # first seq present
        assert f"#{n}</span>" in html                    # last seq present

    def test_no_manifest_and_verify_refuses(self, tmp_path, monkeypatch):
        from tracebi.verify import REFUSED_SNAPSHOT, verify_file
        wb = self._session(tmp_path, monkeypatch)
        out, html = self._export(wb, tmp_path)
        assert list(out.parent.glob("*.manifest.json")) == []
        result = verify_file(html, {})
        assert result["verdict"] == REFUSED_SNAPSHOT
        assert result["ok"] is False and result["exit_code"] == 1

class TestSessionCli:
    """tracebi session export / clear — the CLI over the session feed."""

    def _seed_discovery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wb = discovery_dir(str(tmp_path))
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show("a discovery note")
        write_pins(wb, [{"id": "fig-x", "note": "n", "at_seq": 1}])
        monkeypatch.delenv("TRACEBI_WORKBENCH_DIR")
        return wb

    def test_export_defaults_into_explorations(
            self, tmp_path, monkeypatch, capsys):
        from tracebi import cli
        wb = self._seed_discovery(tmp_path, monkeypatch)
        assert cli.main(["session", "export"]) == 0
        # ONE living record, not a dated diary: the stable name means
        # re-exports overwrite and git carries the timeline.
        out = tmp_path / "explorations" / "discovery.html"
        assert out.is_file()
        assert "a discovery note" in out.read_text(encoding="utf-8")
        assert "verify refuses it by name" in capsys.readouterr().out
        # a second export UPDATES the same file, and says so (heartbeat
        # freshened so the env-less show() posts — a "live server")
        heartbeat(wb)
        show("iterated: second pass")
        assert cli.main(["session", "export"]) == 0
        assert "Updated" in capsys.readouterr().out
        text = out.read_text(encoding="utf-8")
        assert "a discovery note" in text and "second pass" in text
        assert list((tmp_path / "explorations").glob("*.html")) == [out]

    def test_clear_removes_feed_and_pins(self, tmp_path, monkeypatch):
        from tracebi import cli
        wb = self._seed_discovery(tmp_path, monkeypatch)
        assert cli.main(["session", "clear"]) == 0
        assert not os.path.exists(os.path.join(wb, "exhibits.jsonl"))
        assert not os.path.exists(os.path.join(wb, "pins.json"))

    def test_clear_refuses_while_heartbeat_is_fresh(
            self, tmp_path, monkeypatch, capsys):
        from tracebi import cli
        wb = self._seed_discovery(tmp_path, monkeypatch)
        heartbeat(wb)                       # a discovery server is "live"
        assert cli.main(["session", "clear"]) == 1
        assert os.path.exists(os.path.join(wb, "exhibits.jsonl"))
        assert os.path.exists(os.path.join(wb, "pins.json"))
        err = capsys.readouterr().err
        # The refusal names the heartbeat and gives an actionable retry window,
        # not just "stop the server first" (which was un-actionable right after
        # the server was actually stopped).
        assert "heartbeat" in err and "stale" in err


class TestSessionExportMarkdown:
    """The Markdown twin — the git-review format. Raw markdown notes
    verbatim; frames and sketches as pipe tables; same honesty banner."""

    def _session(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show("## Approach\n\nDropped **9** null|mark funds.")
        show(pd.DataFrame({"sector": ["fin|tech"], "fv": [1.5]}),
             chart="bar", x="sector", y="fv", note="sketch")
        write_pins(wb, [{"id": "exhibit-2", "note": "keep", "at_seq": 2}])
        return wb

    def test_md_export_is_reviewable(self, tmp_path, monkeypatch):
        from tracebi._session_export import export_session
        wb = self._session(tmp_path, monkeypatch)
        out = tmp_path / "record.md"
        export_session(wb, str(out), title="Probe record")
        md = out.read_text(encoding="utf-8")
        assert md.startswith("# Probe record")
        assert "lab notebook, not a report" in md
        assert "carry no receipts" in md
        # the note is RAW markdown, verbatim — the format's whole point
        assert "## Approach" in md and "**9**" in md
        assert "**chart sketch** — bar · x: sector · y: fv" in md
        assert "| fin\\|tech | 1.5 |" in md    # pipes escaped in cells
        assert "📌 **pinned:** keep" in md
        assert not (tmp_path / "record.md.manifest.json").exists()

    def test_extension_routes_the_format(self, tmp_path, monkeypatch):
        from tracebi._session_export import export_session
        wb = self._session(tmp_path, monkeypatch)
        export_session(wb, str(tmp_path / "r.html"))
        assert "<!doctype html>" in (tmp_path / "r.html").read_text(
            encoding="utf-8").lower()
        export_session(wb, str(tmp_path / "r.md"))
        assert "<!doctype" not in (tmp_path / "r.md").read_text(
            encoding="utf-8").lower()


class TestPinJoin:
    def test_exhibit_id_beats_at_seq(self, tmp_path, monkeypatch):
        """A pin names its exhibit exactly (exhibit-<n>); at_seq is only the
        fallback for non-exhibit targets like warehouse tables."""
        from tracebi._session_export import export_session
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show("first cell")
        show("second cell")
        write_pins(wb, [{"id": "exhibit-1", "note": "on the FIRST", "at_seq": 2}])
        out = tmp_path / "r.md"
        export_session(wb, str(out))
        md = out.read_text(encoding="utf-8")
        assert md.index("on the FIRST") < md.index("second cell")


def test_dev_server_rejects_dns_rebinding_hosts():
    """The dev server serves the rich /__workbench/state.json; a DNS-rebinding
    page sends its own domain as Host, so only loopback hosts are accepted."""
    from tracebi._dev_server import _host_is_local

    for ok in ["localhost", "localhost:8000", "127.0.0.1", "127.0.0.1:8000",
               "[::1]", "[::1]:8000", ""]:
        assert _host_is_local(ok), f"{ok!r} should be allowed"
    for bad in ["evil.example.com", "evil.example.com:8000", "attacker.test",
                "192.168.1.5:8000", "0.0.0.0:8000"]:
        assert not _host_is_local(bad), f"{bad!r} should be refused"


class TestTimelineAndKeep:
    """The feed says where each exhibit came from and whether a re-run moved
    it; "Keep this" hands the agent a request instead of editing the report."""

    def test_exhibit_records_its_code_and_step(self, tmp_path, monkeypatch):
        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        monkeypatch.chdir(tmp_path)
        script = tmp_path / "transforms" / "probe.py"
        script.parent.mkdir()
        script.write_text(
            "import pandas as pd\n"
            "from tracebi.workbench import show\n"
            "df = pd.DataFrame({'a': [1, 2]})\n"
            "show(df, name='probe')\n")
        exec(compile(script.read_text(), str(script), "exec"), {})
        [ex] = read_exhibits(wb)
        assert ex["step"] == "transform"
        assert ex["origin"]["file"] == os.path.join("transforms", "probe.py")
        assert ex["origin"]["line"] == 4
        assert "show(df, name='probe')" in ex["origin"]["code"]

    def test_reruns_are_marked_new_same_or_changed(self, tmp_path, monkeypatch):
        from tracebi.workbench import _feed

        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show(pd.DataFrame({"v": [1.0]}), name="check")
        show(pd.DataFrame({"v": [1.0]}), name="check")
        show(pd.DataFrame({"v": [2.0]}), name="check")
        changes = [e["change"] for e in sorted(_feed(wb), key=lambda e: e["seq"])]
        assert changes == ["new", "same", "changed"]

    def test_keep_this_pin_carries_a_request_for_the_agent(
            self, wb_model, tmp_path, monkeypatch):
        from tracebi.workbench import write_pins

        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        show(pd.DataFrame({"region": ["NE"], "revenue": [1.0]}),
             name="by_region_check")
        write_pins(wb, [{"id": "keep-1", "kind": "promote", "exhibit": 1,
                         "note": "bar chart under the KPI"}])
        state = collect_state(str(_pkg(tmp_path)), {"wb_model": wb_model})
        [pin] = state["pins"]
        request = pin["request"]
        assert "exhibit #1 'by_region_check'" in request
        assert "report.json" in request and "report.py" in request
        assert "bar chart under the KPI" in request

    def test_a_timeline_message_reaches_the_agent(self, wb_model, tmp_path, monkeypatch):
        from tracebi.workbench import write_pins

        wb = str(tmp_path / "wb")
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", wb)
        write_pins(wb, [{"id": "msg-1", "kind": "message",
                         "note": "split revenue by quarter"}])
        state = collect_state(str(_pkg(tmp_path)), {"wb_model": wb_model})
        [pin] = state["pins"]
        assert pin["request"].endswith(": split revenue by quarter")

    def test_binding_previews_carry_display_text(self, wb_model, tmp_path, monkeypatch):
        monkeypatch.setenv("TRACEBI_WORKBENCH_DIR", str(tmp_path / "wb"))
        state = collect_state(str(_pkg(tmp_path)), {"wb_model": wb_model})
        kpi = next(b for b in state["bindings"] if b["name"] == "kpi")
        assert kpi["preview"] == [{"total": 350.0}]      # raw stays raw
        assert kpi["display"] == [{"total": "350"}]      # as the report writes it
