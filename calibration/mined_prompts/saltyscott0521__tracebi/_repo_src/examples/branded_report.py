"""
A branded, presentation-grade report — without giving up provenance.

The usual assumption is that "governed" and "beautiful" pull against each
other: you either get a rigid corporate template or you hand someone a blank
HTML canvas and lose any claim about where the numbers came from. This example
exists to show that the trade is false, and to be the shape a company can copy.

Everything visual here goes through three seams that already exist:

    Theme               the stylesheet, as data — swap or extend it
    template            a Jinja2 page shell — own the document structure
    section_renderers   add a block type, or replace how a built-in renders

None of those can invent a number. Every figure on the page arrives through a
DataSet, and every DataSet carries the lineage chain that produced it — which
is why the provenance panel at the end is generated, not written. An agent (or
an analyst) composing a report can reach for any of this styling and still
cannot put a figure on the page that the system cannot account for.

That is the whole point: **the design layer is expressive, and the data layer
is closed.**

Run:
    python examples/branded_report.py
    open output/branded_report.html
"""

import sys
from pathlib import Path

# Run this file directly and sys.path[0] is examples/, not the project root, so
# `tracebi` resolves to whatever is pip-installed — which may be a different
# checkout entirely. Prefer the copy this example ships beside.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracebi.model_registry import get_model  # noqa: E402
from tracebi.reports.html_renderer import HTMLRenderer
from tracebi.reports.report import (
    ChartSection, Metric, MetricSection, Report, RowSection, TableSection,
    TextSection,
)
from tracebi.reports.theme import Theme

OUT = Path(__file__).resolve().parent.parent / "output" / "branded_report.html"


# ── 1. The stylesheet, as data ───────────────────────────────────────────────
# Appended to the built-in theme rather than replacing it, so the table,
# chart and metric internals keep working and only the presentation moves.

BRAND_CSS = """
:root {
  --ink:        #10151c;
  --ink-soft:   #5a6675;
  --hairline:   #e6e9ee;
  --surface:    #ffffff;
  --ground:     #f6f7f9;
  --accent:     #0f766e;
  --accent-dim: #d6efec;
  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans:  ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}

body { background: var(--ground); color: var(--ink); font-family: var(--sans);
       -webkit-font-smoothing: antialiased; line-height: 1.55; }

.page { max-width: 1080px; margin: 0 auto; padding: 0 0 72px; }

/* ── Editorial hero ─────────────────────────────────────────────────────── */
.hero { background: linear-gradient(160deg, #0f2b28 0%, #123f39 55%, #0d5f55 100%);
        color: #eaf4f2; padding: 64px 64px 56px; border-radius: 0 0 28px 28px;
        margin-bottom: 44px; position: relative; overflow: hidden; }
.hero::after { content: ""; position: absolute; inset: auto -80px -140px auto;
        width: 380px; height: 380px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,.10), transparent 62%); }
.hero-eyebrow { font-size: 11px; letter-spacing: 2.4px; text-transform: uppercase;
        color: #7fd3c6; font-weight: 700; margin-bottom: 18px; }
.hero-title { font-family: var(--serif); font-size: 50px; line-height: 1.05;
        font-weight: 600; letter-spacing: -.6px; margin: 0 0 14px; max-width: 18ch; }
.hero-standfirst { font-size: 17px; color: #b9d8d2; max-width: 62ch; margin: 0; }
.hero-meta { display: flex; gap: 34px; margin-top: 34px; padding-top: 24px;
        border-top: 1px solid rgba(255,255,255,.16); font-size: 12px; }
.hero-meta div span { display: block; color: #7fd3c6; text-transform: uppercase;
        letter-spacing: 1.2px; font-size: 10px; margin-bottom: 5px; }

/* ── Content rhythm ─────────────────────────────────────────────────────── */
.section { background: var(--surface); border: 1px solid var(--hairline);
        border-radius: 16px; padding: 30px 34px; margin: 0 32px 22px;
        box-shadow: 0 1px 2px rgba(16,21,28,.04), 0 8px 26px rgba(16,21,28,.05); }
.section-title { font-family: var(--serif); font-size: 21px; font-weight: 600;
        letter-spacing: -.2px; color: var(--ink); margin-bottom: 18px; }
.text-heading1 { font-family: var(--serif); font-size: 30px; font-weight: 600;
        letter-spacing: -.4px; margin-bottom: 8px; }
.text-normal { color: var(--ink-soft); font-size: 15px; max-width: 74ch; }

/* ── KPI band ───────────────────────────────────────────────────────────── */
.metric-card { background: var(--surface); border: 1px solid var(--hairline);
        border-radius: 14px; padding: 22px 24px; flex: 1;
        box-shadow: 0 1px 2px rgba(16,21,28,.04); }
.metric-label { font-size: 10.5px; letter-spacing: 1.4px; text-transform: uppercase;
        color: var(--ink-soft); font-weight: 700; margin-bottom: 10px; }
.metric-value { font-family: var(--serif); font-size: 34px; font-weight: 600;
        letter-spacing: -.8px; color: var(--ink); font-variant-numeric: tabular-nums; }

/* ── Tables ─────────────────────────────────────────────────────────────── */
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
thead th { text-align: left; font-size: 10px; letter-spacing: 1.3px;
        text-transform: uppercase; color: var(--ink-soft); font-weight: 700;
        padding: 0 14px 12px; border-bottom: 2px solid var(--ink); }
tbody td { padding: 13px 14px; border-bottom: 1px solid var(--hairline);
        font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: none; }

/* ── Provenance: the audit trail as a design element, not a footnote ────── */
.provenance { margin: 40px 32px 0; background: #0f151c; color: #c9d4dd;
        border-radius: 18px; padding: 34px 38px; }
.provenance h2 { font-family: var(--serif); font-size: 20px; color: #fff;
        margin: 0 0 6px; font-weight: 600; }
.provenance .sub { font-size: 13px; color: #7d8b99; margin-bottom: 24px; }
.prov-chain { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
        margin-bottom: 18px; }
.prov-step { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
        border-radius: 8px; padding: 8px 13px; font-size: 12px; }
.prov-step b { color: #6ee7d7; font-weight: 600; text-transform: uppercase;
        font-size: 10px; letter-spacing: 1px; display: block; margin-bottom: 3px; }
.prov-arrow { color: #46545f; }
.prov-note { font-size: 12px; color: #7d8b99; border-top: 1px solid rgba(255,255,255,.1);
        padding-top: 16px; }
"""


# ── 2. The page shell ────────────────────────────────────────────────────────
# Takes over the document structure. `sections` and `lineage` are rendered by
# the framework and dropped in — the template chooses layout, never content.

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>{{ css }}</style>
</head>
<body>
<div class="page">
  <header class="hero">
    <div class="hero-eyebrow">{{ eyebrow }}</div>
    <h1 class="hero-title">{{ title }}</h1>
    <p class="hero-standfirst">{{ standfirst }}</p>
    <div class="hero-meta">
      <div><span>Prepared by</span>{{ author }}</div>
      <div><span>As at</span>{{ as_at }}</div>
      <div><span>Source</span>{{ source }}</div>
      <div><span>Provenance</span>{{ step_count }} recorded steps</div>
    </div>
  </header>
  {{ sections }}
  {{ provenance }}
</div>
</body>
</html>"""


def build_report():
    model = get_model("wealth_model")

    # ── Every figure below originates in a query against the model, so each
    # ── carries the lineage chain that produced it. Nothing is typed in.
    by_asset_class = model.query(
        fact="fact_holdings",
        measures={"market_value": "sum", "cost_basis": "sum"},
        dimensions=["dim_product.asset_class"],
    )
    by_region = model.query(
        fact="fact_holdings",
        measures={"market_value": "sum"},
        dimensions=["dim_branch.region"],
    )
    by_segment = model.query(
        fact="fact_holdings",
        measures={"market_value": "sum", "units": "sum"},
        dimensions=["dim_client.segment"],
    )

    df = by_asset_class.to_pandas()
    total_aum = float(df["market_value"].sum())
    total_cost = float(df["cost_basis"].sum())
    unrealised = total_aum - total_cost
    classes = int(df["dim_product.asset_class"].nunique())

    report = (
        Report("Assets Under Management")
        .author("Wealth Analytics")
        .description("Positions by asset class, region and client segment.")

        .add(MetricSection(title="Portfolio at a glance", metrics=[
            Metric("Total AUM", total_aum, format="currency0"),
            Metric("Unrealised gain", unrealised, format="currency0",
                   delta=unrealised / total_cost),
            Metric("Asset classes", classes, format="comma"),
        ]))

        .add(TextSection(
            title="Where the money sits",
            content=(
                "Market value is aggregated from individual holdings and joined "
                "to the product, branch and client dimensions. Each figure below "
                "is traceable to the query that produced it — see the provenance "
                "panel at the foot of this page."
            ),
        ))

        # Two blocks side by side. Layout is a first-class section, so an
        # author composes a page without touching markup.
        #
        # column_labels and number_formats are doing the work that separates a
        # presentation artifact from a data dump: without them a reader gets
        # DIM_PRODUCT.ASSET_CLASS as a heading and 1705495.2200000002 as a
        # figure. The vocabulary has always supported this — the defaults are
        # simply raw, which is a trap for anyone (or anything) composing a
        # report without being told to reach for them.
        .add(RowSection(sections=[
            ChartSection(title="AUM by asset class", dataset=by_asset_class,
                         chart_type="bar", x="dim_product.asset_class",
                         y="market_value",
                         xlabel="", ylabel="Market value (USD)",
                         palette=["#0f766e"], show_values=True,
                         figsize=(7.2, 4.2)),
            TableSection(title="By region", dataset=by_region,
                         column_labels={"dim_branch.region": "Region",
                                        "market_value": "Market value"},
                         number_formats={"market_value": "currency0"},
                         totals=["market_value"]),
        ], widths=[0.56, 0.44]))

        .add(TableSection(title="By client segment", dataset=by_segment,
                          column_labels={"dim_client.segment": "Segment",
                                         "market_value": "Market value",
                                         "units": "Units held"},
                          number_formats={"market_value": "currency0",
                                          "units": "comma"},
                          color_scale={"market_value": "#0f766e"},
                          totals=["market_value", "units"]))
    )
    return report, [by_asset_class, by_region, by_segment]


def distinct_lineage(datasets) -> list:
    """Unique lineage steps across every dataset on the page, in order."""
    seen, chain = set(), []
    for ds in datasets:
        for node in ds.lineage:
            key = (node.operation, str(node.description))
            if key not in seen:
                seen.add(key)
                chain.append(node)
    return chain


def render_provenance(datasets) -> str:
    """
    The audit trail, rendered as a feature rather than an appendix.

    Generated from the lineage the framework recorded while the queries ran —
    which is exactly why it cannot be faked or forgotten. Nobody wrote these
    steps; they are what actually happened.
    """
    # Deduplicated: the three queries share their loads and joins, and showing
    # "Loaded 'holdings'" three times would misrepresent the work as well as
    # reading badly. The hero's step count reads the same helper, so the two
    # numbers cannot drift apart.
    chain = distinct_lineage(datasets)

    steps = '<span class="prov-arrow">→</span>'.join(
        f'<div class="prov-step"><b>{n.operation}</b>{n.description}</div>'
        for n in chain
    )
    return f"""
    <section class="provenance">
      <h2>How these numbers were produced</h2>
      <div class="sub">Recorded automatically as each query ran.</div>
      <div class="prov-chain">{steps}</div>
      <div class="prov-note">
        {len(chain)} steps across {len(datasets)} datasets. This panel is
        generated from the lineage chain attached to each figure — it is not
        written by the report's author, and it cannot be omitted.
      </div>
    </section>"""


def main():
    report, datasets = build_report()
    step_count = len(distinct_lineage(datasets))

    renderer = HTMLRenderer(
        theme=Theme.default().with_overrides(BRAND_CSS, name="editorial"),
        template=SHELL,
        template_context={
            "eyebrow": "Quarterly portfolio review",
            "standfirst": (
                "A governed report: styled freely, sourced strictly. Every "
                "figure carries the chain of operations that produced it."
            ),
            "author": "Wealth Analytics",
            "as_at": "Q3 2026",
            "source": "WealthModel",
            "step_count": step_count,
            "provenance": render_provenance(datasets),
        },
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(report, str(OUT))
    print(f"Wrote {OUT}")
    print(f"  {len(report.sections)} sections, {step_count} lineage steps recorded")


if __name__ == "__main__":
    main()
