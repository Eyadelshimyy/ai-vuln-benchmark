"""
An interactive report with no JavaScript at all.

Two questions this answers.

**Can a report be interactive?** Yes — tabs, drill-downs, sticky headers and
hover detail are all reachable with HTML and CSS alone. Verified rather than
claimed: the check at the end greps the rendered file and fails if a single
``<script>`` or inline handler made it in.

**Should an agent be able to add JavaScript?** It can — the shipped interactive
runtime lets an agent author its own ``script.js`` — but under a locked honesty
rule the runtime enforces: **controls subset which stamped rows a figure
displays; they never compute a new number.** A dropdown may narrow a table to
one region; it may not sum that region into a fresh figure — **a filtered KPI
needs its own binding**, its own stamped query, not a client-side
recomputation. The old fear is real but defused: a script *can* read the DOM and
rewrite a number, yet it cannot make that number read as verified, because a
figure is verified only while a stamped query backs it. Interactivity that
subsets is safe; computation that invents has nowhere to hide. So the line is
not "no JS" — it is "no new numbers".

Which leaves a happy arrangement. Interactivity is a property of the **theme
and its section renderers** — authored once, reviewed, reusable — and any
report, however it was composed, inherits it. The author picks a theme; they do
not implement behaviour.

This example also shows the `section_renderers` seam doing real work: `TabsSection`
is a section type the framework has never heard of, rendered by a callable
passed to the renderer.

Run:
    python examples/interactive_report.py
    open output/interactive_report.html
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracebi.model_registry import get_model  # noqa: E402
from tracebi.reports.html_renderer import HTMLRenderer  # noqa: E402
from tracebi.reports.report import (  # noqa: E402
    Metric, MetricSection, Report, ReportSection, TableSection, TextSection,
)
from tracebi.reports.theme import Theme  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output" / "interactive_report.html"


# ── A section type the framework does not ship ──────────────────────────────
# The renderer dispatches on section_type and consults section_renderers first,
# so a new block type needs no change to TraceBi itself.

@dataclass
class TabsSection(ReportSection):
    """Several tables behind CSS-only tabs. Each pane keeps its own lineage."""
    panes: list = field(default_factory=list)   # [(label, TableSection, DataSet)]

    def __post_init__(self):
        self.section_type = "tabs"


CSS = """
:root {
  --ink:#101418; --soft:#5d6975; --line:#e4e8ec; --card:#fff; --ground:#f5f7f8;
  --accent:#1d4ed8; --accent-soft:#eef2ff;
}
body { background:var(--ground); color:var(--ink); line-height:1.55;
       font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.page { max-width:1040px; margin:0 auto; padding:0 0 64px; }

.masthead { background:var(--card); border-bottom:1px solid var(--line);
       padding:44px 40px 30px; margin-bottom:26px; }
.masthead h1 { font-size:32px; letter-spacing:-.5px; margin:0 0 8px; }
.masthead p { color:var(--soft); margin:0; max-width:70ch; }
.badge { display:inline-block; background:var(--accent-soft); color:var(--accent);
       font-size:11px; font-weight:700; letter-spacing:1.4px; text-transform:uppercase;
       padding:5px 11px; border-radius:20px; margin-bottom:14px; }

.section { background:var(--card); border:1px solid var(--line); border-radius:14px;
       padding:26px 30px; margin:0 28px 20px;
       box-shadow:0 1px 2px rgba(16,20,24,.04), 0 6px 20px rgba(16,20,24,.04); }
.section-title { font-size:18px; font-weight:650; margin-bottom:16px; }
.metric-card { background:var(--card); border:1px solid var(--line);
       border-radius:12px; padding:18px 20px; flex:1; }
.metric-label { font-size:10px; letter-spacing:1.3px; text-transform:uppercase;
       color:var(--soft); font-weight:700; margin-bottom:8px; }
.metric-value { font-size:28px; font-weight:650; font-variant-numeric:tabular-nums; }

/* ── Tabs, with a radio input as the state ──────────────────────────────── */
/* The checked radio is the only "state" involved; sibling selectors do the   */
/* showing and hiding. No script, and it keeps working with JS disabled.      */
.tabs input[type=radio] { position:absolute; opacity:0; pointer-events:none; }
.tab-strip { display:flex; gap:4px; border-bottom:1px solid var(--line);
       margin-bottom:18px; }
.tab-strip label { padding:9px 16px; font-size:13px; font-weight:600;
       color:var(--soft); cursor:pointer; border-bottom:2px solid transparent;
       margin-bottom:-1px; transition:color .15s, border-color .15s; }
.tab-strip label:hover { color:var(--ink); }
.tab-pane { display:none; }
.tabs #t0:checked ~ .tab-strip label[for=t0],
.tabs #t1:checked ~ .tab-strip label[for=t1],
.tabs #t2:checked ~ .tab-strip label[for=t2] {
       color:var(--accent); border-bottom-color:var(--accent); }
.tabs #t0:checked ~ .panes .tab-pane:nth-of-type(1),
.tabs #t1:checked ~ .panes .tab-pane:nth-of-type(2),
.tabs #t2:checked ~ .panes .tab-pane:nth-of-type(3) { display:block; }

/* ── Drill-down: native <details>, no script ────────────────────────────── */
details.prov { margin-top:16px; border-top:1px dashed var(--line); padding-top:14px; }
details.prov summary { cursor:pointer; font-size:12px; font-weight:600;
       color:var(--accent); list-style:none; }
details.prov summary::-webkit-details-marker { display:none; }
details.prov summary::before { content:"▸ "; }
details.prov[open] summary::before { content:"▾ "; }
details.prov ol { margin:12px 0 0 0; padding-left:18px; color:var(--soft);
       font-size:12.5px; }
details.prov li { margin-bottom:6px; }
details.prov code { background:var(--accent-soft); color:var(--accent);
       padding:1px 6px; border-radius:4px; font-size:11px; font-weight:600; }

/* ── Tables: sticky head, hover, tabular figures ────────────────────────── */
.pane-scroll { max-height:340px; overflow:auto; border:1px solid var(--line);
       border-radius:10px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
thead th { position:sticky; top:0; background:var(--card); z-index:1;
       text-align:left; font-size:10px; letter-spacing:1.2px; text-transform:uppercase;
       color:var(--soft); font-weight:700; padding:12px 14px;
       border-bottom:2px solid var(--ink); }
tbody td { padding:11px 14px; border-bottom:1px solid var(--line);
       font-variant-numeric:tabular-nums; }
tbody tr:hover { background:var(--accent-soft); }

@media print {
  .tab-pane { display:block !important; }   /* print shows every pane */
  details.prov[open] summary::before { content:""; }
  .section { break-inside:avoid; box-shadow:none; }
}
"""


def lineage_details(ds, label: str) -> str:
    """The chain behind one pane, collapsed behind a native disclosure."""
    steps = "".join(
        f"<li><code>{n.operation}</code> {n.description}</li>" for n in ds.lineage
    )
    return (
        f'<details class="prov"><summary>How “{label}” was computed '
        f'({len(ds.lineage)} steps)</summary><ol>{steps}</ol></details>'
    )


def render_tabs(section: TabsSection) -> str:
    """
    Renderer for the custom `tabs` type.

    Emits arbitrary HTML — which is fine, because this is engineer-authored
    code living beside the theme, not something a spec can express. The tables
    inside are still rendered by the framework, so their figures and lineage
    are untouched.
    """
    inner = HTMLRenderer()
    radios, strip, panes = "", "", ""
    for i, (label, table, ds) in enumerate(section.panes):
        checked = " checked" if i == 0 else ""
        radios += f'<input type="radio" name="tabs" id="t{i}"{checked}>'
        strip += f'<label for="t{i}">{label}</label>'
        panes += (
            f'<div class="tab-pane"><div class="pane-scroll">'
            f"{inner._render_table(table)}</div>{lineage_details(ds, label)}</div>"
        )
    title = f'<div class="section-title">{section.title}</div>' if section.title else ""
    return (
        f'<div class="section"><div class="tabs">{title}{radios}'
        f'<div class="tab-strip">{strip}</div>'
        f'<div class="panes">{panes}</div></div></div>'
    )


def build():
    model = get_model("wealth_model")

    cuts = [
        ("By region",      "dim_branch.region",      "Region"),
        ("By segment",     "dim_client.segment",     "Segment"),
        ("By asset class", "dim_product.asset_class", "Asset class"),
    ]
    panes, datasets = [], []
    for label, dim, human in cuts:
        ds = model.query(fact="fact_holdings",
                         measures={"market_value": "sum", "units": "sum"},
                         dimensions=[dim])
        panes.append((label, TableSection(
            dataset=ds,
            column_labels={dim: human, "market_value": "Market value",
                           "units": "Units"},
            number_formats={"market_value": "currency0", "units": "comma"},
            totals=["market_value", "units"],
        ), ds))
        datasets.append(ds)

    total = float(datasets[0].to_pandas()["market_value"].sum())

    report = (
        Report("Holdings Review")
        .author("Wealth Analytics")
        .add(MetricSection(title="Position summary", metrics=[
            Metric("Total AUM", total, format="currency0"),
            Metric("Cuts available", len(cuts), format="comma"),
            Metric("Lineage steps", sum(len(d.lineage) for d in datasets),
                   format="comma"),
        ]))
        .add(TextSection(
            title="Reading this report",
            content=(
                "Switch cuts with the tabs below; each keeps its own provenance, "
                "expandable underneath. Nothing here runs JavaScript — the tabs "
                "are a checked radio and a sibling selector, and the drill-downs "
                "are native disclosure elements. Printing expands every pane."
            ),
        ))
        .add(TabsSection(title="Holdings by cut", panes=panes))
    )
    return report


def main():
    renderer = HTMLRenderer(
        theme=Theme.default().with_overrides(CSS, name="interactive"),
        section_renderers={"tabs": render_tabs},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(build(), str(OUT))

    html = OUT.read_text()
    scripts = re.findall(r"<script\b", html, re.I)
    handlers = re.findall(r'\son[a-z]+\s*=\s*["\']', html, re.I)
    print(f"Wrote {OUT}")
    print(f"  <script> tags: {len(scripts)}   inline handlers: {len(handlers)}")
    if scripts or handlers:
        raise SystemExit("FAILED: this report is supposed to contain no JavaScript")
    print("  verified: interactive, and not one line of JavaScript")


if __name__ == "__main__":
    main()
