"""The analyst knowledge base is well-formed AND actually delivered.

The failure mode this guards is the one the map found in docs/agents/sop-*.md:
genuine practice that no agent surface points at, so it rots unread. These tests
assert the lessons are well-formed and — the load-bearing part — that they reach
an agent through both consumption forms (the `tracebi context` payload and the
`tracebi-analyst` skill). If the delivery is ever silently dropped, CI fails.
"""

import re
from pathlib import Path

from tracebi.knowledge import get_lesson, index, list_lessons

_SKILL = (Path(__file__).resolve().parents[1]
          / "skills" / "tracebi-analyst" / "SKILL.md")


def test_lessons_parse_and_slugs_are_unique():
    lessons = list_lessons()
    assert lessons, "no lessons found — the curriculum is empty"
    slugs = [ls.slug for ls in lessons]
    assert len(slugs) == len(set(slugs)), f"duplicate lesson slugs: {slugs}"
    for ls in lessons:
        assert ls.title and ls.title != ls.slug, f"{ls.slug}: missing title"
        assert ls.when, f"{ls.slug}: missing 'when' trigger — an agent needs it"
        assert len(ls.body) > 100, f"{ls.slug}: body too thin to teach anything"


def test_cross_links_resolve():
    """Every [[slug]] reference points at a real lesson (link liberally, but not
    to nothing — a dead reference teaches the agent a lesson that isn't there)."""
    slugs = {ls.slug for ls in list_lessons()}
    for ls in list_lessons():
        for ref in re.findall(r"\[\[([a-z0-9-]+)\]\]", ls.body):
            assert ref in slugs, f"{ls.slug} links [[{ref}]] which does not exist"


def test_the_curriculum_is_delivered_in_context_both_tiers():
    """The lessons must appear in describe() — brief AND full — or a bring-your-
    own agent over MCP never learns the curriculum exists. This is the guard
    against the SOP-doc fate: a folder nobody's surface points at."""
    from tracebi.capabilities import describe

    for brief in (True, False):
        payload = describe(brief=brief)
        block = payload.get("analyst_knowledge")
        assert block, f"analyst_knowledge missing from context (brief={brief})"
        got = {l["slug"] for l in block["lessons"]}
        assert got == {ls.slug for ls in list_lessons()}, (
            f"context curriculum out of sync with the lessons (brief={brief})")
        assert "tracebi knowledge" in block["fetch"]
        assert block["fetch"].index("lessons") < block["fetch"].index(
            "tracebi knowledge")
        rules = " ".join(payload["conventions"]["rules"])
        assert rules.index("list_models") < rules.index("tracebi validate")
        assert "list_reports" in rules
        assert payload["warehouse"]["when"].startswith("Call describe_table")


def test_context_index_matches_the_files():
    idx = index()
    assert {i["slug"] for i in idx} == {ls.slug for ls in list_lessons()}
    for i in idx:
        assert i["title"] and i["when"]


def test_skill_points_at_the_knowledge_base():
    """The skill must delegate to the lessons (single source of truth), not
    re-state the guidance where it can drift."""
    text = _SKILL.read_text(encoding="utf-8")
    assert "tracebi knowledge" in text, "skill does not reference the lessons"
    # The seed lessons the skill names by slug must exist.
    for slug in ("ratio-of-totals", "weighted-vs-plain-mean", "grain-and-fanout",
                 "verify-your-own-work"):
        assert slug in text, f"skill names '{slug}' in prose"
        assert get_lesson(slug) is not None, f"skill names missing lesson '{slug}'"


_DESIGNER = _SKILL.parents[1] / "tracebi-designer" / "SKILL.md"
_REPO = _SKILL.parents[2]


def test_every_design_lesson_reaches_the_designer_skill_and_both_guides():
    """The design lessons share the analyst curriculum's delivery, so the same
    rot applies: a design lesson no skill or guide names is one no agent
    reads. Every `design-` lesson must be named in the designer skill and in
    the repo AGENTS.md, and the skill must name no lesson that doesn't exist.
    The scaffolded guide must at least point at the curriculum."""
    design = {ls.slug for ls in list_lessons() if ls.slug.startswith("design-")}
    assert len(design) >= 5, "the design curriculum is missing"

    skill = _DESIGNER.read_text(encoding="utf-8")
    assert "tracebi knowledge" in skill, "designer skill does not reference the lessons"
    named = set(re.findall(r"\bdesign-[a-z-]+[a-z]", skill))
    assert design <= named, f"designer skill omits {sorted(design - named)}"
    assert named <= design, f"designer skill names missing lessons {sorted(named - design)}"

    agents = (_REPO / "AGENTS.md").read_text(encoding="utf-8")
    missing = sorted(s for s in design if s not in agents)
    assert not missing, f"AGENTS.md does not name {missing}"

    from tracebi.cli import _INIT_AGENTS_MD
    assert "tracebi knowledge design-" in _INIT_AGENTS_MD, (
        "the scaffolded AGENTS.md never points at the design lessons")


def test_every_measure_kind_and_aggregation_is_in_the_vocabulary():
    """A feature the vocabulary doesn't publish effectively doesn't exist for
    an agent. Derived from the code, so a new measure kind or aggregation
    that isn't documented fails here instead of per-feature look-alikes."""
    import inspect

    from tracebi.capabilities import describe
    from tracebi.model.data_model import _AGG_FUNCS, DataModel

    sm = describe()["semantic_model"]
    published = {k["kind"] for k in sm["measure_kinds"]}
    renamed = {"column": "simple", "expr": "expression"}
    params = inspect.signature(DataModel.add_measure).parameters
    kinds = {renamed.get(p, p) for p in params
             if p in ("column", "expr", "ratio", "share", "rank", "running",
                      "period_end", "offset", "growth", "to_date")}
    assert len(kinds) == 10, "a measure kind was added: list it above"
    assert kinds <= published, f"undocumented measure kinds: {kinds - published}"
    assert set(_AGG_FUNCS) <= set(sm["aggregations"])
    assert any("percentile" in a for a in sm["aggregations"])
    assert "add_value_bins" in sm["value_bins"]["declare"]
    assert "or" in str(sm["filter_forms"])


def test_reference_model_obeys_the_weighted_mean_lesson():
    """The 'would the agent do the analysis RIGHT' rot-guard the map asked for.
    A measure declared agg='mean' but described 'weighted' is the exact silent-
    wrong trap weighted-vs-plain-mean names — and the reference model shipped it
    once. If it ever comes back (or lands in the scaffold), fail here, because
    the canonical example teaches by being correct."""
    model_src = (Path(__file__).resolve().parents[1] / "examples"
                 / "portfolio_project" / "models" / "portfolio_model.yaml")
    import yaml

    # Parsing drops comments, so a note ABOUT the trap ("previously agg=mean,
    # 'weighted'") doesn't trip the lint; only real declarations count.
    doc = yaml.safe_load(model_src.read_text(encoding="utf-8"))
    for measure in doc["measures"]:
        is_mean = measure.get("agg") in ("mean", "avg")
        says_weighted = re.search(r"weight", str(measure.get("description", "")), re.IGNORECASE)
        assert not (is_mean and says_weighted), (
            "portfolio_model declares a 'weighted' measure as a plain mean — "
            "see tracebi knowledge weighted-vs-plain-mean; use a ratio of "
            "sum(value*weight) / sum(weight) instead.")


# ── The guardrail that teaches — the fanout raise's sibling ────────────────────
# A guardrail is worth more than a paragraph of prose because it fires at the
# moment of the mistake and no model can skip it. This one refuses a mean of a
# rate/ratio and cites the lesson.

class TestMeanOfARatioGuard:
    def _model(self):
        from tracebi import DataModel
        return DataModel("t")

    def test_refuses_additive_aggregation_of_a_rate(self):
        import pytest
        # mean AND sum of a rate are both refused; min/max are not.
        for agg in ("mean", "avg", "sum"):
            for name, col in [("avg_margin_pct", "margin_pct"), ("yield", "y"),
                              ("total_spread_bps", "spread_bps")]:
                with pytest.raises(ValueError, match="aggregates a rate"):
                    self._model().add_measure(name, column=col, agg=agg)

    def test_min_max_of_a_rate_are_allowed(self):
        # The widest spread / lowest yield are legitimate.
        self._model().add_measure("max_spread_bps", column="spread_bps", agg="max")
        self._model().add_measure("min_yield", column="yield", agg="min")

    def test_summing_a_weighted_numerator_is_allowed(self):
        # Σ(value × weight) is the CORRECT building block of a weighted average —
        # "weighted" in the description must NOT refuse a sum (only a mean).
        self._model().add_measure(
            "spread_x_par", expr="spread_bps * par", agg="sum",
            description="Σ(spread × par) — par-weighted-spread numerator")

    def test_refuses_a_weighted_mean_declared_as_a_plain_mean(self):
        import pytest
        with pytest.raises(ValueError, match="aggregates a rate"):
            self._model().add_measure("wtd", column="spread", agg="mean",
                                      description="Weighted average spread")

    def test_the_refusal_teaches(self):
        import pytest
        with pytest.raises(ValueError) as exc:
            self._model().add_measure("avg_yield", column="yield", agg="mean")
        msg = str(exc.value)
        assert "ratio-of-totals" in msg          # cites the lesson
        assert "allow_rate_agg=True" in msg      # names the escape hatch
        assert "sum(numerator)" in msg           # gives the correct pattern

    def test_does_not_false_positive_on_additive_means(self):
        # Legitimate means of additive quantities must still work.
        for name, col in [("avg_order_value", "revenue"), ("avg_age", "age"),
                          ("mean_qty", "quantity")]:
            self._model().add_measure(name, column=col, agg="mean")

    def test_allow_rate_agg_is_the_escape_hatch(self):
        # Explicit opt-out, like allow_fanout — never blocks a deliberate choice.
        self._model().add_measure("avg_pct", column="p", agg="mean",
                                   allow_rate_agg=True)
        self._model().add_measure("sum_pct", column="p2", agg="sum",
                                   allow_rate_agg=True)

    def test_guard_is_documented_in_the_vocabulary(self):
        from tracebi.capabilities import describe
        constraints = describe()["semantic_model"]["constraints"]
        assert any("allow_rate_agg" in c for c in constraints), (
            "the mean-of-a-ratio guard must be in the vocabulary so an agent "
            "knows it exists and knows the escape hatch")


class TestValueBasedRateGuard:
    """The name guard is blind to a per-row ratio whose column name matches no
    token (mark_cost = fair_value / cost). At execution the values are here, so
    an additive aggregation of a ratio-shaped column is refused — the mark_cost
    hole a round-3 field test drove a mean-of-8.488 straight through."""

    def _model(self, n=40):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        # mark_cost ~1.0, fractional, with one 316,000x warrant in the tail —
        # the exact shape (median ≈ 1) the guard keys on.
        mark = [round(0.9 + 0.005 * i, 4) for i in range(n)]
        mark[0] = 316000.0
        fact = pd.DataFrame({
            "hid": range(n), "fund_id": [1] * n, "mark_cost": mark,
            "fair_value": [1_000_000.0 + i for i in range(n)],
            "cost": [900_000.0 + i for i in range(n)],
        })
        dim = pd.DataFrame({"fund_id": [1], "fund": ["A"]})
        m = DataModel("m")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim": dim}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim", connector="mem", source="dim")
        m.add_dimension("dim", table_name="dim", key_col="fund_id",
                        attributes=["fund"])
        m.add_fact("fact", table_name="fact",
                   measures=["mark_cost", "fair_value", "cost"],
                   foreign_keys={"dim": "fund_id"})
        return m

    def test_refuses_mean_and_sum_of_a_ratio_shaped_column(self):
        import pytest
        for agg in ("mean", "sum", "avg"):
            with pytest.raises(ValueError, match="per-row ratios"):
                self._model().query(fact="fact", measures={"mc": ("mark_cost", agg)})

    def test_min_max_of_the_same_column_are_allowed(self):
        # extremes of a ratio are legitimate — the guard only refuses additive aggs.
        m = self._model()
        m.query(fact="fact", measures={"hi": ("mark_cost", "max")})
        m.query(fact="fact", measures={"lo": ("mark_cost", "min")})

    def test_the_ratio_of_totals_is_the_correct_form_and_is_allowed(self):
        m = self._model()
        m.add_measure("fv", column="fair_value", agg="sum")
        m.add_measure("cb", column="cost", agg="sum")
        m.add_measure("mark", ratio=("fv", "cb"))
        df = m.query(fact="fact", measures=["fv", "cb", "mark"]).to_pandas()
        assert df["mark"].iloc[0] == df["fv"].iloc[0] / df["cb"].iloc[0]

    def test_query_flag_is_the_escape(self):
        df = self._model().query(fact="fact", measures={"mc": ("mark_cost", "mean")},
                                 allow_rate_agg=True).to_pandas()
        assert df["mc"].iloc[0] > 1.0   # the (wrong) mean, explicitly permitted

    def test_declared_measure_allow_rate_agg_persists_to_query_time(self):
        m = self._model()
        m.add_measure("avg_mark", column="mark_cost", agg="mean", allow_rate_agg=True)
        m.query(fact="fact", measures=["avg_mark"])   # no raise

    def test_money_columns_are_not_tripped(self):
        # median in the millions — nowhere near 1.0, so the guard stays silent.
        self._model().query(fact="fact", measures={"fv": ("fair_value", "sum")})

    def test_small_fixtures_do_not_trip(self):
        # below the row-count gate — a per-row-ratio problem matters at scale.
        self._model(n=10).query(fact="fact", measures={"mc": ("mark_cost", "mean")})

    def test_the_refusal_teaches(self):
        import pytest
        with pytest.raises(ValueError) as exc:
            self._model().query(fact="fact", measures={"mc": ("mark_cost", "mean")})
        msg = str(exc.value)
        assert "ratio-of-totals" in msg
        assert "allow_rate_agg=True" in msg
        assert "sum(numerator)" in msg

    def test_integer_counts_stored_as_float_are_not_tripped(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        # whole numbers near 1 (a 0/1/2 flag as float) are not ratios.
        fact = pd.DataFrame({"hid": range(40), "flag": [float(i % 2) for i in range(40)]})
        m = DataModel("f")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_fact("fact", table_name="fact", measures=["flag"])
        m.query(fact="fact", measures={"flags": ("flag", "sum")})   # no raise


# ── Expressiveness pulled back into the governed lane: distribution aggs ───────
# A mean hides skew; median/stddev used to force report.py (ungoverned, no
# receipt, no guard). Now they are declarative measures — governed, verifiable,
# deterministic — paired with the summarize-a-distribution lesson.

class TestDistributionAggregations:
    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        fact = pd.DataFrame({"desk_id": [1, 1, 1, 1, 2, 2, 2, 2],
                             "pnl": [10, 12, 11, 500, 9, 11, 10, 12]})
        dim = pd.DataFrame({"desk_id": [1, 2], "desk": ["A", "B"]})
        m = DataModel("risk")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim_desk": dim}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_desk", connector="mem", source="dim_desk")
        m.add_dimension("dim_desk", table_name="dim_desk", key_col="desk_id",
                        attributes=["desk"])
        m.add_fact("f", table_name="fact", measures=["pnl"],
                   foreign_keys={"dim_desk": "desk_id"})
        m.add_measure("mean_pnl", column="pnl", agg="mean")
        m.add_measure("median_pnl", column="pnl", agg="median")
        m.add_measure("stddev_pnl", column="pnl", agg="stddev")
        m.connect()
        return m

    def _run(self, m):
        from tracebi.model.data_model import QuerySpec
        return m.execute(QuerySpec.from_dict(
            {"fact": "f", "measures": ["mean_pnl", "median_pnl", "stddev_pnl"],
             "dimensions": ["dim_desk.desk"]})).to_pandas()

    def test_median_is_robust_where_the_mean_is_dragged_by_an_outlier(self):
        row = self._run(self._model()).set_index("dim_desk.desk").loc["A"]
        assert row["mean_pnl"] > 100        # dragged by the 500 outlier
        assert 11 <= row["median_pnl"] <= 12   # the honest centre
        assert row["stddev_pnl"] > 100      # the spread flags the skew

    def test_median_of_a_rate_is_not_refused(self):
        # The rate guard targets ADDITIVE aggs (sum/mean); a median or stddev of
        # a rate is a legitimate summary and must stay allowed.
        from tracebi import DataModel
        DataModel("t").add_measure("median_spread_bps", column="spread_bps",
                                   agg="median")


class TestPercentiles:
    """p<N> percentile aggregations — the tail a mean and even a median hide
    (worst marks, largest drawdowns). p50 is the median; any p0–p100."""

    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        fact = pd.DataFrame({"hid": range(100), "fund_id": [1] * 100,
                             "mark": [float(i) for i in range(1, 101)]})
        dim = pd.DataFrame({"fund_id": [1], "fund": ["A"]})
        m = DataModel("p")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim": dim}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim", connector="mem", source="dim")
        m.add_dimension("dim", table_name="dim", key_col="fund_id",
                        attributes=["fund"])
        m.add_fact("fact", table_name="fact", measures=["mark"],
                   foreign_keys={"dim": "fund_id"})
        m.add_measure("p90_mark", column="mark", agg="p90")
        m.connect()
        return m

    def test_percentiles_match_the_interpolated_quantile(self):
        import numpy as np
        vals = [float(i) for i in range(1, 101)]
        m = self._model()
        # declared p90 and ad-hoc p50/p95/p99 all match numpy's linear interp.
        df = m.query(fact="fact", measures=["p90_mark"]).to_pandas()
        assert df["p90_mark"].iloc[0] == np.percentile(vals, 90)
        adhoc = m.query(fact="fact", measures={
            "p50": ("mark", "p50"), "p95": ("mark", "p95"),
            "p99": ("mark", "p99")}).to_pandas()
        assert adhoc["p50"].iloc[0] == np.percentile(vals, 50)
        assert adhoc["p95"].iloc[0] == np.percentile(vals, 95)
        assert adhoc["p99"].iloc[0] == np.percentile(vals, 99)

    def test_out_of_range_percentile_is_refused(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="unsupported aggregation"):
            DataModel("t").add_measure("bad", column="x", agg="p150")

    def test_a_non_percentile_p_name_is_refused(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="unsupported aggregation"):
            DataModel("t").add_measure("bad", column="x", agg="pinky")

# ── Share-of-total: a report.py computation pulled into the governed lane ──────

class TestShareMeasure:
    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        fact = pd.DataFrame({"region_id": [1, 2, 3], "rev": [600.0, 300.0, 100.0]})
        dim = pd.DataFrame({"region_id": [1, 2, 3],
                            "region": ["West", "East", "North"]})
        m = DataModel("s")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim_region": dim}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_region", connector="mem", source="dim_region")
        m.add_dimension("dim_region", table_name="dim_region", key_col="region_id",
                        attributes=["region"])
        m.add_fact("f", table_name="fact", measures=["rev"],
                   foreign_keys={"dim_region": "region_id"})
        m.add_measure("revenue", column="rev", agg="sum")
        m.add_measure("revenue_share", share="revenue", format="percent")
        m.connect()
        return m

    def _run(self, m, **extra):
        from tracebi.model.data_model import QuerySpec
        q = {"fact": "f", "measures": ["revenue", "revenue_share"],
             "dimensions": ["dim_region.region"], **extra}
        return m.execute(QuerySpec.from_dict(q))

    def test_shares_are_the_fraction_of_the_total(self):
        df = self._run(self._model()).to_pandas().set_index("dim_region.region")
        assert df.loc["West", "revenue_share"] == 0.6
        assert df.loc["East", "revenue_share"] == 0.3
        assert round(df["revenue_share"].sum(), 6) == 1.0

    def test_share_takes_no_agg(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="take no agg"):
            DataModel("t").add_measure("s", share="revenue", agg="sum")

class TestRankAndRunning:
    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        fact = pd.DataFrame({"rid": [1, 2, 3, 4], "rev": [500.0, 300.0, 150.0, 50.0]})
        dim = pd.DataFrame({"rid": [1, 2, 3, 4],
                            "region": ["West", "East", "North", "South"]})
        m = DataModel("c")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim_region": dim}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_region", connector="mem", source="dim_region")
        m.add_dimension("dim_region", table_name="dim_region", key_col="rid",
                        attributes=["region"])
        m.add_fact("f", table_name="fact", measures=["rev"],
                   foreign_keys={"dim_region": "rid"})
        m.add_measure("revenue", column="rev", agg="sum")
        m.add_measure("revenue_share", share="revenue", format="percent")
        m.add_measure("rev_rank", rank="revenue")
        m.add_measure("cum_share", running="revenue_share", format="percent")
        m.connect()
        return m

    def _run(self, m):
        from tracebi.model.data_model import QuerySpec
        return m.execute(QuerySpec.from_dict({
            "fact": "f",
            "measures": ["revenue", "rev_rank", "revenue_share", "cum_share"],
            "dimensions": ["dim_region.region"],
            "order_by": [{"column": "rev_rank", "desc": False}]}))

    def test_the_full_concentration_table_is_governed(self):
        import pytest
        df = self._run(self._model()).to_pandas()
        assert df["rev_rank"].tolist() == [1, 2, 3, 4]          # largest first
        assert df["revenue_share"].tolist() == pytest.approx([0.5, 0.3, 0.15, 0.05])
        assert df["cum_share"].tolist() == pytest.approx([0.5, 0.8, 0.95, 1.0])

    def test_windows_are_deterministic(self):
        m = self._model()
        assert self._run(m).fingerprint() == self._run(m).fingerprint()

    def test_rank_and_running_take_no_agg(self):
        import pytest
        from tracebi import DataModel
        for kw in ({"rank": "revenue"}, {"running": "revenue"}):
            with pytest.raises(ValueError, match="take no agg"):
                DataModel("t").add_measure("w", agg="sum", **kw)

class TestPerGroupTopN:
    """A partitioned rank restarts per group; with a having on it, that is
    top-N-per-group — which a global order_by+limit cannot express."""

    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        fv = {("Tech", "A"): 100, ("Tech", "B"): 90, ("Tech", "C"): 80,
              ("Fin", "E"): 200, ("Fin", "F"): 150, ("Fin", "G"): 50}
        rows, dims, i = [], [], 0
        for (sec, iss), v in fv.items():
            i += 1
            rows.append({"hid": i, "issuer_id": i, "fair_value": float(v)})
            dims.append({"issuer_id": i, "issuer": iss, "sector": sec})
        m = DataModel("m")
        m.add_connector(MemoryConnector("mem", tables={
            "fact": pd.DataFrame(rows), "dim_issuer": pd.DataFrame(dims)}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_issuer", connector="mem", source="dim_issuer")
        m.add_dimension("dim_issuer", table_name="dim_issuer", key_col="issuer_id",
                        attributes=["issuer", "sector"])
        m.add_fact("fact", table_name="fact", measures=["fair_value"],
                   foreign_keys={"dim_issuer": "issuer_id"})
        m.add_measure("fv", column="fair_value", agg="sum")
        m.add_measure("rank_in_sector", rank="fv",
                      partition_by="dim_issuer.sector")
        m.add_measure("global_rank", rank="fv")
        m.connect()
        return m

    def _rows(self, m, **extra):
        q = dict(fact="fact", measures=["fv", "rank_in_sector"],
                 dimensions=["dim_issuer.sector", "dim_issuer.issuer"], **extra)
        return m.query(**q).to_pandas()

    def test_rank_restarts_per_partition(self):
        df = self._rows(self._model())
        got = {(r["dim_issuer.sector"], r["dim_issuer.issuer"]): r["rank_in_sector"]
               for _, r in df.iterrows()}
        assert got[("Fin", "E")] == 1 and got[("Fin", "G")] == 3
        assert got[("Tech", "A")] == 1 and got[("Tech", "C")] == 3

    def test_global_rank_and_partitioned_rank_coexist(self):
        df = self._model().query(
            fact="fact", measures=["fv", "rank_in_sector", "global_rank"],
            dimensions=["dim_issuer.sector", "dim_issuer.issuer"]).to_pandas()
        g = dict(zip(df["dim_issuer.issuer"], df["global_rank"]))
        assert g["E"] == 1 and g["G"] == 6          # global across both sectors

    def test_having_on_the_partitioned_rank_is_top_n_per_group(self):
        df = self._rows(self._model(), having={"rank_in_sector": {"lte": 2}},
                        order_by=["dim_issuer.sector", "rank_in_sector"])
        assert list(df["dim_issuer.issuer"]) == ["E", "F", "A", "B"]

    def test_partitioned_rank_is_deterministic(self):
        m = self._model()
        assert self._rows(m).equals(self._rows(m))

    def test_partition_column_must_be_a_query_dimension(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="partition_by references"):
            m.query(fact="fact", measures=["fv", "rank_in_sector"],
                    dimensions=["dim_issuer.issuer"])   # sector not selected

    def test_partition_by_only_on_rank_or_running(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="only applies to rank and running"):
            DataModel("t").add_measure("s", share="revenue",
                                       partition_by="dim_x.y")

    def test_partition_by_needs_dotted_refs(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="dim_name.attribute"):
            DataModel("t").add_measure("r", rank="revenue", partition_by="sector")

    def test_partition_is_documented_in_the_vocabulary(self):
        from tracebi.capabilities import describe
        kinds = {k["kind"]: k for k in describe()["semantic_model"]["measure_kinds"]}
        assert "partition_by" in kinds["rank"]["args"]
        assert "top-n-per-group" in kinds["rank"]["note"].lower()

class TestTimeGrain:
    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        dates = pd.DataFrame({"d": [1, 2, 3, 4], "order_date": pd.to_datetime(
            ["2024-01-15", "2024-01-20", "2024-02-10", "2024-03-05"])})
        fact = pd.DataFrame({"date_id": [1, 2, 3, 4], "rev": [100.0, 200.0, 300.0, 400.0]})
        m = DataModel("t")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim_date": dates}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_date", connector="mem", source="dim_date")
        m.add_dimension("dim_date", table_name="dim_date", key_col="d",
                        attributes=["order_date"])
        m.add_time_grain("dim_date", "order_month", source="order_date", grain="month")
        m.add_fact("f", table_name="fact", measures=["rev"],
                   foreign_keys={"dim_date": "date_id"})
        m.add_measure("revenue", column="rev", agg="sum")
        m.connect()
        return m

    def _by_month(self, m):
        from tracebi.model.data_model import QuerySpec
        return m.execute(QuerySpec.from_dict(
            {"fact": "f", "measures": ["revenue"],
             "dimensions": ["dim_date.order_month"]}))

    def test_group_by_month_rolls_up_governed(self):
        df = self._by_month(self._model()).to_pandas()
        assert df["revenue"].tolist() == [300.0, 300.0, 400.0]   # Jan/Feb/Mar

    def test_bad_grain_is_refused(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="not supported"):
            m.add_time_grain("dim_date", "x", source="order_date", grain="fortnight")

    def test_grain_is_discoverable_in_model_info(self):
        info = self._model().info()
        dim = next(d for d in info["dimensions"] if d["name"] == "dim_date")
        assert dim["derived"]["order_month"] == {
            "kind": "date_trunc", "of": "order_date", "grain": "month"}

class TestValueBins:
    """Value bins: group by a BAND of a numeric dimension column — a governed
    CASE over ranges, instead of a pre-baked bucket or report.py."""

    def _model(self, **bins):
        import numpy as np
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        cust = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                             "score": [550.0, 650.0, 720.0, 830.0, 700.0, np.nan]})
        fact = pd.DataFrame({"lid": range(1, 7), "cust_id": [1, 2, 3, 4, 5, 6],
                             "amt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        m = DataModel("m")
        m.add_connector(MemoryConnector("mem", tables={
            "fact": fact, "dim_customer": cust}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_customer", connector="mem", source="dim_customer")
        m.add_dimension("dim_customer", table_name="dim_customer", key_col="id",
                        attributes=["score"])
        m.add_value_bins("dim_customer", "score_band", source="score",
                         edges=bins.get("edges", [600, 700, 800]),
                         labels=bins.get("labels"))
        m.add_fact("fact", table_name="fact", measures=["amt"],
                   foreign_keys={"dim_customer": "cust_id"})
        m.add_measure("total", column="amt", agg="sum")
        m.connect()
        return m

    def _by_band(self, m):
        return {r["dim_customer.score_band"]: r["total"] for _, r in
                m.query(fact="fact", measures=["total"],
                        dimensions=["dim_customer.score_band"]).to_pandas().iterrows()}

    def test_values_land_in_the_right_band(self):
        got = self._by_band(self._model())
        assert got["< 600"] == 10.0        # 550
        assert got["600–700"] == 20.0      # 650
        assert got["700–800"] == 80.0      # 720 + 700
        assert got["≥ 800"] == 40.0        # 830

    def test_null_source_groups_as_null_not_the_top_band(self):
        got = self._by_band(self._model())
        assert got.get("≥ 800") == 40.0    # the NULL row (60) did NOT land here
        assert 60.0 in got.values()        # it is its own (NULL) group

    def test_custom_labels(self):
        m = self._model(edges=[700], labels=["subprime", "prime"])
        got = self._by_band(m)
        # < 700: 550,650,700? no — 700 is NOT < 700, so prime. subprime: 550,650.
        assert got["subprime"] == 30.0     # 10 + 20
        assert got["prime"] == 120.0       # 30 + 40 + 50 (720,830,700)

    def test_edges_must_be_increasing_and_nonempty(self):
        import pytest
        from tracebi import DataModel, MemoryConnector
        import pandas as pd
        m = DataModel("t")
        m.add_connector(MemoryConnector("mem", tables={
            "d": pd.DataFrame({"k": [1], "x": [1.0]})}))
        m.add_table("d", connector="mem", source="d")
        m.add_dimension("d", table_name="d", key_col="k", attributes=["x"])
        with pytest.raises(ValueError, match="at least one edge"):
            m.add_value_bins("d", "b", source="x", edges=[])
        with pytest.raises(ValueError, match="strictly increasing"):
            m.add_value_bins("d", "b", source="x", edges=[700, 600])
        with pytest.raises(ValueError, match="one label per band"):
            m.add_value_bins("d", "b", source="x", edges=[600, 700], labels=["a"])

    def test_bins_are_discoverable_in_model_info(self):
        info = self._model().info()
        dim = next(d for d in info["dimensions"] if d["name"] == "dim_customer")
        band = dim["derived"]["score_band"]
        assert band["kind"] == "bin" and band["of"] == "score"
        assert band["bands"] == ["< 600", "600–700", "700–800", "≥ 800"]

class TestSemiAdditive:
    """period_end (semi-additive) measures: sum a stock across dimensions but
    take the latest snapshot over time — never the AUM double-count."""

    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        # Two funds, two month-end snapshots each. balance is a point-in-time
        # stock: snapshot 1 (Jan 31), snapshot 2 (Feb 29).
        fact = pd.DataFrame({
            "holding_id": range(1, 9),
            "fund_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "as_of_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "balance": [100.0, 50.0, 200.0, 20.0, 120.0, 60.0, 190.0, 30.0],
        })
        dim_fund = pd.DataFrame({"fund_id": [1, 2], "fund": ["Alpha", "Beta"]})
        dim_date = pd.DataFrame({
            "as_of_id": [1, 2],
            "as_of_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "month": ["2024-01", "2024-02"],
        })
        m = DataModel("sa")
        m.add_connector(MemoryConnector("mem", tables={
            "fact": fact, "dim_fund": dim_fund, "dim_date": dim_date}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_fund", connector="mem", source="dim_fund")
        m.add_table("dim_date", connector="mem", source="dim_date")
        m.add_dimension("dim_fund", table_name="dim_fund", key_col="fund_id",
                        attributes=["fund"])
        m.add_dimension("dim_date", table_name="dim_date", key_col="as_of_id",
                        attributes=["as_of_date", "month"])
        m.add_fact("fact", table_name="fact", measures=["balance"],
                   foreign_keys={"dim_fund": "fund_id", "dim_date": "as_of_id"})
        m.add_measure("aum", period_end=("balance", "dim_date.as_of_date"),
                      description="Period-end AUM")
        m.connect()
        return m

    def _q(self, m, **kw):
        return m.query(fact="fact", measures=["aum"], **kw).to_pandas()

    def test_by_dimension_takes_each_group_latest_snapshot(self):
        # Latest snapshot is Feb 29: Alpha 120+60=180, Beta 190+30=220.
        df = self._q(self._model(), dimensions=["dim_fund.fund"])
        got = dict(zip(df["dim_fund.fund"], df["aum"]))
        assert got == {"Alpha": 180.0, "Beta": 220.0}

    def test_by_time_grain_is_a_real_period_end_series_not_a_sum(self):
        # Each month takes its own latest snapshot — NOT Jan+Feb summed.
        df = self._q(self._model(), dimensions=["dim_date.month"])
        got = dict(zip(df["dim_date.month"], df["aum"]))
        assert got == {"2024-01": 370.0, "2024-02": 400.0}

    def test_total_is_the_latest_snapshot_not_every_snapshot_summed(self):
        # The trap: naive sum(balance) = 770 (both snapshots). period_end = 400.
        df = self._q(self._model())
        assert df["aum"].iloc[0] == 400.0

    def test_a_filter_applies_before_the_period_end(self):
        df = self._q(self._model(), filters={"dim_fund.fund": "Alpha"})
        assert df["aum"].iloc[0] == 180.0

    def test_a_ratio_can_reference_a_period_end_measure(self):
        m = self._model()
        m.add_measure("cost_basis", column="balance", agg="sum",
                      allow_additive=True)   # stand-in denominator
        m.add_measure("aum_over_base", ratio=("aum", "cost_basis"))
        df = m.query(fact="fact", measures=["aum", "cost_basis", "aum_over_base"],
                     dimensions=["dim_fund.fund"]).to_pandas()
        # Alpha: aum 180 / base 330 (100+50+120+60); ratio computed on totals.
        row = df[df["dim_fund.fund"] == "Alpha"].iloc[0]
        assert row["aum_over_base"] == row["aum"] / row["cost_basis"]

    def test_period_end_takes_no_agg(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="take no agg"):
            m.add_measure("bad", period_end=("balance", "dim_date.as_of_date"),
                          agg="sum")

    def test_period_end_date_must_be_a_dim_attribute(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="dim_name.attribute"):
            m.add_measure("bad", period_end=("balance", "as_of_date"))

    def test_aggregate_false_is_refused(self):
        import pytest
        from tracebi.model.data_model import QuerySpec
        m = self._model()
        with pytest.raises(ValueError, match="aggregate=False"):
            m.execute(QuerySpec.from_dict(
                {"fact": "fact", "measures": ["aum"], "aggregate": False}))

class TestStockSumGuard:
    """Summing a stock-named measure double-counts across snapshots — refused
    at declaration, with a genuine-flow escape."""

    def _model(self):
        from tracebi import DataModel, MemoryConnector
        import pandas as pd
        m = DataModel("g")
        m.add_connector(MemoryConnector("mem", tables={
            "fact": pd.DataFrame({"k": [1], "balance": [1.0]})}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_fact("fact", table_name="fact", measures=["balance"])
        return m

    def test_refuses_a_plain_sum_of_a_stock(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="point-in-time stock"):
            m.add_measure("aum", column="balance", agg="sum")

    def test_the_refusal_points_at_period_end(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="period_end"):
            m.add_measure("total_nav", column="balance", agg="sum")

    def test_mean_of_a_stock_is_allowed(self):
        # Average balance is a legitimate metric — only sum double-counts.
        self._model().add_measure("avg_balance", column="balance", agg="mean")

    def test_allow_additive_is_the_escape_for_a_genuine_flow(self):
        self._model().add_measure("balance_change", column="balance", agg="sum",
                                  allow_additive=True)

    def test_a_non_stock_sum_is_untouched(self):
        self._model().add_measure("headcountish", column="balance", agg="sum",
                                  allow_additive=True)
        # a name with no stock token sums freely
        self._model().add_measure("revenue", column="balance", agg="sum")

    def test_guard_is_documented_in_the_vocabulary(self):
        from tracebi.capabilities import describe
        text = str(describe()["semantic_model"]["constraints"])
        assert "stock" in text and "allow_additive" in text


class TestBooleanOrFilters:
    """OR across different columns is a filter group, not two queries. filters
    AND-s by default; the reserved 'or'/'and' keys build a boolean tree."""

    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        fact = pd.DataFrame({"id": range(1, 7), "iid": [1, 2, 3, 4, 5, 6],
                             "amt": [10., 20, 30, 40, 50, 60],
                             "status": ["a", "a", "b", "b", "a", "b"]})
        dim = pd.DataFrame({"iid": [1, 2, 3, 4, 5, 6],
                            "sector": ["Tech", "Fin", "Tech", "Energy", "Fin", "Tech"],
                            "rating": ["AAA", "BBB", "BBB", "AAA", "AAA", "CCC"]})
        m = DataModel("m")
        m.add_connector(MemoryConnector("mem", tables={"fact": fact, "dim": dim}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim", connector="mem", source="dim")
        m.add_dimension("dim", table_name="dim", key_col="iid",
                        attributes=["sector", "rating"])
        m.add_fact("fact", table_name="fact", measures=["amt"],
                   foreign_keys={"dim": "iid"})
        m.add_measure("n", column="id", agg="count")
        m.connect()
        return m

    def _ids(self, m, filters):
        # the set of matching row ids, via a count over a dimension we can invert
        df = m.query(fact="fact", measures=["n"], filters=filters).to_pandas()
        return int(df["n"].iloc[0])

    def test_plain_filters_still_and(self):
        m = self._model()
        assert self._ids(m, {"status": "a", "dim.sector": "Tech"}) == 1  # only id 1

    def test_or_across_different_columns(self):
        m = self._model()
        # Tech {1,3,6} OR AAA {1,4,5} = {1,3,4,5,6}
        assert self._ids(m, {"or": [{"dim.sector": "Tech"},
                                    {"dim.rating": "AAA"}]}) == 5

    def test_and_of_a_leaf_with_an_or_group(self):
        m = self._model()
        # status=a {1,2,5} AND (Tech OR AAA) {1,3,4,5,6} = {1,5}
        assert self._ids(m, {"status": "a",
                             "or": [{"dim.sector": "Tech"},
                                    {"dim.rating": "AAA"}]}) == 2

    def test_nested_or_of_and_with_operators(self):
        m = self._model()
        # (amt>=50) {5,6} OR (Tech AND BBB) {3} = {3,5,6}
        assert self._ids(m, {"or": [{"amt": {"gte": 50}},
                                    {"and": [{"dim.sector": "Tech"},
                                             {"dim.rating": "BBB"}]}]}) == 3

    def test_malformed_or_groups_are_refused(self):
        import pytest
        m = self._model()
        for bad in ({"or": {"x": 1}}, {"or": []}, {"or": ["nope"]}):
            with pytest.raises(ValueError):
                m.query(fact="fact", measures=["n"], filters=bad)

    def test_a_bad_column_inside_or_is_validated(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="not found"):
            m.query(fact="fact", measures=["n"],
                    filters={"or": [{"badcol": 1}, {"dim.sector": "Tech"}]})

    def test_spec_validate_recurses_into_or(self):
        from tracebi.model.data_model import QuerySpec
        m = self._model()
        errs, _ = m.check_query_spec(QuerySpec.from_dict({
            "fact": "fact", "measures": ["n"],
            "filters": {"or": [{"dim.sector": "Tech"}, {"dim.nope": "x"}]}}))
        assert any("nope" in e[1] for e in errs)

class TestPeriodOverPeriod:
    """offset/growth measures: year-over-year and friends as governed measures
    over a declared time grain, not a self-join in report.py."""

    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        dates = pd.date_range("2023-01-01", "2024-03-01", freq="MS")
        rows, dim_rows, did = [], [], 0
        for d in dates:
            did += 1
            dim_rows.append({"dt_id": did, "d": d})
            rows.append({"id": did, "dt_id": did,
                         "rev": 1000.0 + (d.year - 2023) * 1200 + d.month * 100})
        m = DataModel("m")
        m.add_connector(MemoryConnector("mem", tables={
            "fact": pd.DataFrame(rows), "dim_date": pd.DataFrame(dim_rows)}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_date", connector="mem", source="dim_date")
        m.add_dimension("dim_date", table_name="dim_date", key_col="dt_id",
                        attributes=["d"])
        m.add_time_grain("dim_date", "month", source="d", grain="month")
        m.add_fact("fact", table_name="fact", measures=["rev"],
                   foreign_keys={"dim_date": "dt_id"})
        m.add_measure("revenue", column="rev", agg="sum")
        m.add_measure("rev_ly", offset=("revenue", "year", 1))
        m.add_measure("rev_yoy", growth=("revenue", "year", 1), format="percent")
        m.add_measure("rev_mom", growth=("revenue", "month", 1))
        m.connect()
        return m

    def _by_month(self, m, measures):
        import pandas as pd
        df = m.query(fact="fact", measures=measures,
                     dimensions=["dim_date.month"]).to_pandas()
        df["m"] = pd.to_datetime(df["dim_date.month"]).dt.strftime("%Y-%m")
        return df.set_index("m")

    def test_offset_reads_the_prior_year_value(self):
        df = self._by_month(self._model(), ["revenue", "rev_ly"])
        # 2024-01 revenue 2300 (1000+1200+100); prior = 2023-01 = 1100.
        assert df.loc["2024-01", "revenue"] == 2300.0
        assert df.loc["2024-01", "rev_ly"] == 1100.0

    def test_offset_is_null_when_no_prior_period_is_present(self):
        df = self._by_month(self._model(), ["revenue", "rev_ly"])
        assert df.loc["2023-01", "rev_ly"] != df.loc["2023-01", "rev_ly"]  # NaN

    def test_growth_is_current_minus_prior_over_prior(self):
        df = self._by_month(self._model(), ["revenue", "rev_yoy"])
        assert df.loc["2024-01", "rev_yoy"] == (2300.0 - 1100.0) / 1100.0

    def test_month_over_month(self):
        df = self._by_month(self._model(), ["revenue", "rev_mom"])
        # 2023-02 (1200) vs 2023-01 (1100)
        assert df.loc["2023-02", "rev_mom"] == (1200.0 - 1100.0) / 1100.0

    def test_limit_after_the_shift_keeps_the_comparison(self):
        # order_by+limit runs AFTER the pop, so "latest month" keeps its YoY.
        m = self._model()
        df = m.query(fact="fact", measures=["revenue", "rev_yoy"],
                     dimensions=["dim_date.month"],
                     order_by=["-dim_date.month"], limit=1).to_pandas()
        assert df["rev_yoy"].iloc[0] == (2500.0 - 1300.0) / 1300.0   # 2024-03

    def test_needs_a_time_grain_in_the_query(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="declared time grain"):
            m.query(fact="fact", measures=["revenue", "rev_yoy"])

    def test_offset_takes_no_agg_and_needs_a_triple(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="take no agg"):
            DataModel("t").add_measure("x", offset=("revenue", "year", 1),
                                       agg="sum")
        with pytest.raises(ValueError, match="triple"):
            DataModel("t").add_measure("x", offset=("revenue", "year"))

    def test_bad_unit_and_bad_n_are_refused(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="not.*supported|unit"):
            DataModel("t").add_measure("x", offset=("revenue", "fortnight", 1))
        with pytest.raises(ValueError, match="positive integer"):
            DataModel("t").add_measure("x", growth=("revenue", "year", 0))

class TestToDate:
    """to_date measures: YTD/QTD as governed cumulatives that reset on a period
    boundary, not a hand-rolled running sum in report.py."""

    def _model(self):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        dates = pd.date_range("2023-01-01", "2024-03-01", freq="MS")
        rows, dim_rows, did = [], [], 0
        for d in dates:
            did += 1
            dim_rows.append({"dt_id": did, "d": d})
            rows.append({"id": did, "dt_id": did, "rev": 100.0})
        m = DataModel("m")
        m.add_connector(MemoryConnector("mem", tables={
            "fact": pd.DataFrame(rows), "dim_date": pd.DataFrame(dim_rows)}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_table("dim_date", connector="mem", source="dim_date")
        m.add_dimension("dim_date", table_name="dim_date", key_col="dt_id",
                        attributes=["d"])
        m.add_time_grain("dim_date", "month", source="d", grain="month")
        m.add_fact("fact", table_name="fact", measures=["rev"],
                   foreign_keys={"dim_date": "dt_id"})
        m.add_measure("revenue", column="rev", agg="sum")
        m.add_measure("rev_ytd", to_date=("revenue", "year"))
        m.add_measure("rev_qtd", to_date=("revenue", "quarter"))
        m.connect()
        return m

    def _by_month(self, m, measures):
        import pandas as pd
        df = m.query(fact="fact", measures=measures,
                     dimensions=["dim_date.month"]).to_pandas()
        df["m"] = pd.to_datetime(df["dim_date.month"]).dt.strftime("%Y-%m")
        return df.set_index("m")

    def test_ytd_accumulates_and_resets_each_year(self):
        df = self._by_month(self._model(), ["revenue", "rev_ytd"])
        assert df.loc["2023-03", "rev_ytd"] == 300.0    # Jan+Feb+Mar
        assert df.loc["2023-12", "rev_ytd"] == 1200.0   # full year
        assert df.loc["2024-01", "rev_ytd"] == 100.0    # reset

    def test_qtd_resets_each_quarter(self):
        df = self._by_month(self._model(), ["revenue", "rev_qtd"])
        assert df.loc["2023-03", "rev_qtd"] == 300.0    # end of Q1
        assert df.loc["2023-04", "rev_qtd"] == 100.0    # start of Q2

    def test_reset_period_must_be_coarser_than_the_grain(self):
        import pytest
        m = self._model()
        m.add_measure("rev_mtd", to_date=("revenue", "month"))
        with pytest.raises(ValueError, match="coarser"):
            m.query(fact="fact", measures=["revenue", "rev_mtd"],
                    dimensions=["dim_date.month"])

    def test_needs_a_time_grain_in_the_query(self):
        import pytest
        m = self._model()
        with pytest.raises(ValueError, match="time grain"):
            m.query(fact="fact", measures=["revenue", "rev_ytd"])

    def test_to_date_takes_no_agg_and_needs_a_pair(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="take no agg"):
            DataModel("t").add_measure("x", to_date=("revenue", "year"), agg="sum")
        with pytest.raises(ValueError, match="pair"):
            DataModel("t").add_measure("x", to_date=("revenue",))

    def test_bad_period_is_refused(self):
        import pytest
        from tracebi import DataModel
        with pytest.raises(ValueError, match="not.*supported|period"):
            DataModel("t").add_measure("x", to_date=("revenue", "fortnight"))
