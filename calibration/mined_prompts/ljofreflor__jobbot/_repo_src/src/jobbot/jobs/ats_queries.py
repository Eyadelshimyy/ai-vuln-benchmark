"""ATS host search-query templates — conditions of possibility for board discovery.

Phenomenology (not a wish-list feature): high-signal openings often *appear* on
ATS-hosted boards (Ashby / Greenhouse / Lever) before aggregators. That
appearance is possible when you constrain a web search to those hosts plus
remote/geo tokens. JobBot already classifies those hosts (``portals/detect``);
it does **not** query a search engine itself (see ``companies discover
--search-results``). This module only emits the query strings so a human can
run them and feed URLs back.

Rejected compression of the viral tip: Kickresume-style “AI rewrite to beat
ATS” — facts stay in ``profile.yaml``; never invent experience.
"""

from __future__ import annotations

from dataclasses import dataclass

from jobbot.portals.detect import AtsKind

# Hosts that surface employer postings (site: operator targets).
ATS_SEARCH_HOSTS: dict[AtsKind, str] = {
    AtsKind.ASHBY: "jobs.ashbyhq.com",
    AtsKind.GREENHOUSE: "greenhouse.io",
    AtsKind.LEVER: "lever.co",
}

DEFAULT_ATS_KINDS: tuple[AtsKind, ...] = (
    AtsKind.ASHBY,
    AtsKind.GREENHOUSE,
    AtsKind.LEVER,
)


@dataclass(frozen=True)
class AtsSearchQuery:
    """One human-runnable web search query aimed at an ATS host."""

    kind: AtsKind
    host: str
    query: str
    regions: tuple[str, ...]
    remote: bool


def build_ats_search_queries(
    *,
    kinds: tuple[AtsKind, ...] | None = None,
    regions: tuple[str, ...] = (),
    remote: bool = True,
    keywords: tuple[str, ...] = (),
) -> list[AtsSearchQuery]:
    """Build ``site:{host} …`` queries. Empty regions → host + remote/keywords only."""
    selected = kinds or DEFAULT_ATS_KINDS
    out: list[AtsSearchQuery] = []
    for kind in selected:
        host = ATS_SEARCH_HOSTS.get(kind)
        if not host:
            continue
        parts = [f"site:{host}"]
        if remote:
            parts.append('"remote"')
        for region in regions:
            token = region.strip()
            if token:
                parts.append(f'"{token}"')
        for word in keywords:
            token = word.strip()
            if token:
                parts.append(f'"{token}"' if " " in token else token)
        out.append(
            AtsSearchQuery(
                kind=kind,
                host=host,
                query=" ".join(parts),
                regions=regions,
                remote=remote,
            )
        )
    return out


def format_queries_help(queries: list[AtsSearchQuery]) -> str:
    """Human-facing next steps after printing queries."""
    lines = [
        "Run these in a web search engine (JobBot does not query one itself).",
        "Paste posting / board URLs into a YAML for:",
        "  jobbot companies discover SEEDS.yaml --search-results hits.yaml",
        "Or ingest a JD you opened: jobbot jobs add --file path/to/jd.txt",
        "Do not outsource facts to an external “ATS rewrite” tool — profile.yaml is SoT.",
        "",
        "Queries:",
    ]
    for q in queries:
        # Escape brackets so Rich does not treat [ashby] as markup.
        lines.append(f"  ({q.kind.value}) {q.query}")
    return "\n".join(lines)
