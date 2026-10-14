"""
LineageDiagram — visualise a DataSet or Report lineage as a directed graph.

Nodes are color-coded by operation type:

    load      → navy (#003366)
    bronze    → bronze (#CD7F32)
    silver    → silver (#C0C0C0)
    gold      → gold (#FFD700)
    filter    → green (#2E7D32)
    transform → amber (#F59E0B)
    join      → orange (#E65100)
    sort      → purple (#6A1B9A)
    select    → steel (#37474F)
    rename    → teal (#00695C)
    <other>   → grey (#757575)

Requires: networkx, matplotlib (for .show() and .to_html())
"""

from __future__ import annotations

import html
import io
import textwrap
from typing import TYPE_CHECKING, Union

from tracebi.model.dataset import DataSet, LineageNode

if TYPE_CHECKING:
    import networkx as nx

    from tracebi.reports.report import Report


_OP_COLORS: dict[str, str] = {
    "load":         "#003366",
    "bronze":       "#CD7F32",
    "silver":       "#C0C0C0",
    "gold":         "#FFD700",
    "landing":      "#4A90E2",   # blue
    "manipulation": "#7B68EE",   # slate-blue
    "final":        "#10B981",   # emerald
    "filter":       "#2E7D32",
    "transform":    "#F59E0B",
    "join":         "#E65100",
    "sort":         "#6A1B9A",
    "select":       "#37474F",
    "rename":       "#00695C",
    "warning":      "#D97706",   # warm amber for non-blocking warnings
}
_DEFAULT_COLOR = "#757575"


def _require_networkx():
    """
    Import networkx on demand.

    Kept out of module scope so ``import tracebi`` works on the base install
    (pandas only) and ``to_mermaid()`` — which needs no graph library — stays
    usable without the optional extras.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised via extras
        raise ImportError(
            "Graph rendering requires networkx. "
            "Install with: pip install 'tracebi[lineage]'. "
            "(LineageDiagram.to_mermaid() works without it.)"
        ) from exc
    return nx


def _node_color(operation: str) -> str:
    return _OP_COLORS.get(operation.lower(), _DEFAULT_COLOR)


def _text_color(bg: str) -> str:
    """Return black or white text depending on background brightness."""
    r = int(bg[1:3], 16)
    g = int(bg[3:5], 16)
    b = int(bg[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#FFFFFF"


class LineageDiagram:
    """
    Directed-graph visualisation of a DataSet or Report lineage.

    Usage::

        from tracebi.lineage.diagram import LineageDiagram

        # From a DataSet
        ds = orders_ds.filter("status == 'shipped'")
        diag = LineageDiagram(ds)

        diag.show()            # matplotlib window / Jupyter inline
        diag.to_html("lineage.html")
        print(diag.to_mermaid())

        # From a Report (aggregates lineage across all sections)
        diag = LineageDiagram(report)
    """

    def __init__(self, source: Union["DataSet", "Report"]) -> None:
        """
        Args:
            source: A :class:`~tracebi.model.dataset.DataSet` or a
                    :class:`~tracebi.reports.report.Report`.  For a Report,
                    lineage is collected from every section that exposes a
                    DataSet.
        """
        self._nodes: list[LineageNode] = []
        self._title: str = ""

        # Import here to avoid circular deps
        try:
            from tracebi.reports.report import Report
            if isinstance(source, Report):
                self._title = source.name
                self._nodes = self._collect_report_lineage(source)
                return
        except ImportError:
            pass

        if isinstance(source, DataSet):
            self._title = source.name
            self._nodes = list(source.lineage)
        else:
            raise TypeError(
                f"LineageDiagram accepts a DataSet or Report, got {type(source).__name__}"
            )

    def _collect_report_lineage(self, report) -> list[LineageNode]:
        seen_ids: set[int] = set()
        nodes: list[LineageNode] = []
        for section in report.data_sections():
            ds = getattr(section, "dataset", None)
            if ds is None:
                continue
            for node in ds.lineage:
                nid = id(node)
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    nodes.append(node)
        return nodes

    # ── Graph construction ─────────────────────────────────────

    def _build_graph(self) -> "nx.DiGraph":
        nx = _require_networkx()
        G = nx.DiGraph()
        for i, node in enumerate(self._nodes):
            label = f"[{node.operation.upper()}]\n{textwrap.shorten(node.description, 40)}"
            G.add_node(i, label=label, operation=node.operation, node=node)
            if i > 0:
                G.add_edge(i - 1, i)
        return G

    # ── Public API ─────────────────────────────────────────────

    def show(self) -> None:
        """
        Render the lineage graph inline (Jupyter) or in a matplotlib window.

        Requires ``matplotlib``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        nx = _require_networkx()
        G = self._build_graph()
        if not G.nodes:
            print("No lineage nodes to display.")
            return

        pos = nx.spring_layout(G, seed=42, k=2.5) if len(G) > 1 else {0: (0, 0)}
        colors = [_node_color(G.nodes[n]["operation"]) for n in G.nodes]

        fig, ax = plt.subplots(figsize=(max(8, len(G) * 1.5), 5))
        ax.set_title(f"Lineage — {self._title}", fontsize=13, pad=12)
        ax.axis("off")

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999",
                               arrows=True, arrowsize=15)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                               node_size=2200, node_shape="s")

        for n in G.nodes:
            x, y = pos[n]
            bg = _node_color(G.nodes[n]["operation"])
            fg = _text_color(bg)
            label = G.nodes[n]["label"]
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=7, color=fg, wrap=True,
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                              edgecolor="none", alpha=0.9))

        # Legend
        legend_ops = sorted(set(G.nodes[n]["operation"] for n in G.nodes))
        patches = [mpatches.Patch(color=_node_color(op), label=op) for op in legend_ops]
        ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.7)

        plt.tight_layout()
        plt.show()

    def to_html(self, path: str) -> None:
        """
        Save a standalone HTML file with an embedded SVG lineage diagram.

        Args:
            path: Output file path (e.g. ``"lineage.html"``).
        """
        svg = self._render_svg()
        title = html.escape(self._title)
        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lineage — {title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #f5f5f5;
            display: flex; flex-direction: column; align-items: center;
            padding: 2rem; }}
    h1   {{ font-size: 1.4rem; color: #333; margin-bottom: 1rem; }}
    .svg-wrap {{ background: white; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,.12); padding: 1.5rem; }}
    svg  {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Lineage &mdash; {title}</h1>
  <div class="svg-wrap">{svg}</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"Lineage diagram saved to {path}")

    def to_mermaid(self) -> str:
        """
        Return a Mermaid flowchart markdown string for the lineage graph.

        Paste into any Mermaid-compatible renderer or GitHub markdown
        (```mermaid ... ```) to get an interactive diagram.
        """
        if not self._nodes:
            return "graph LR\n  empty[No lineage nodes]"

        lines = ["graph LR"]
        for i, node in enumerate(self._nodes):
            desc = node.description.replace('"', "'")
            short = textwrap.shorten(desc, 50)
            op = node.operation.upper()
            lines.append(f'  N{i}["{op}: {short}"]')
            if i > 0:
                lines.append(f"  N{i-1} --> N{i}")

        # Style nodes
        for i, node in enumerate(self._nodes):
            color = _node_color(node.operation)
            text = _text_color(color)
            lines.append(f"  style N{i} fill:{color},color:{text},stroke:none")

        return "\n".join(lines)

    # ── SVG helpers ────────────────────────────────────────────

    def _render_svg(self) -> str:
        """Render the graph to an SVG string using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        nx = _require_networkx()
        G = self._build_graph()
        if not G.nodes:
            return "<svg><text x='10' y='20'>No lineage nodes.</text></svg>"

        pos = nx.spring_layout(G, seed=42, k=2.5) if len(G) > 1 else {0: (0, 0)}
        colors = [_node_color(G.nodes[n]["operation"]) for n in G.nodes]

        fig, ax = plt.subplots(figsize=(max(8, len(G) * 1.5), 5))
        ax.set_title(f"Lineage — {self._title}", fontsize=13, pad=12)
        ax.axis("off")

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999",
                               arrows=True, arrowsize=15)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                               node_size=2200, node_shape="s")
        for n in G.nodes:
            x, y = pos[n]
            bg = _node_color(G.nodes[n]["operation"])
            fg = _text_color(bg)
            ax.text(x, y, G.nodes[n]["label"], ha="center", va="center",
                    fontsize=7, color=fg, multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                              edgecolor="none", alpha=0.9))

        legend_ops = sorted(set(G.nodes[n]["operation"] for n in G.nodes))
        patches = [mpatches.Patch(color=_node_color(op), label=op) for op in legend_ops]
        ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.7)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read().decode("utf-8")

    def __repr__(self) -> str:
        return (
            f"<LineageDiagram title={self._title!r} "
            f"nodes={len(self._nodes)}>"
        )
