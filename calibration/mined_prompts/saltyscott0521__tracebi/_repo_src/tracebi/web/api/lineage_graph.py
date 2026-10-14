"""
Shared lineage-graph helpers for API routers.

Converts a flat list of LineageNode dicts into React Flow nodes + edges.
Used by the reports and dashboards routers (and any future lineage view)
so operation colors stay consistent across pages.

Join steps record ``right_chain_len`` in their metadata — the number of
trailing pre-join lineage nodes that belong to the right side. When present,
the graph branches: both parent chains flow into the join node. Lineage
recorded before this convention renders as a linear chain, as before.
"""

from __future__ import annotations

OP_COLORS: dict[str, str] = {
    "load":         "#003366",
    "filter":       "#2E7D32",
    "transform":    "#F59E0B",
    "join":         "#E65100",
    "aggregate":    "#AD1457",
    "assign":       "#4338CA",
    "sort":         "#6A1B9A",
    "select":       "#37474F",
    "rename":       "#00695C",
    "bronze":       "#CD7F32",
    "silver":       "#C0C0C0",
    "gold":         "#FFD700",
    "landing":      "#4A90E2",
    "manipulation": "#7B68EE",
    "final":        "#10B981",
    "warning":      "#D97706",
}
DEFAULT_COLOR = "#757575"

X_GAP = 240
Y_GAP = 130


def lineage_to_graph(nodes: list[dict]) -> dict:
    """Convert a flat lineage list to React Flow nodes + edges."""
    rf_nodes: list[dict] = []
    rf_edges: list[dict] = []
    lanes = {"max": 0}  # lane allocator so parallel join branches never overlap

    def emit(node: dict, depth: int, lane: int) -> str:
        nid = str(len(rf_nodes))
        color = OP_COLORS.get(node["operation"].lower(), DEFAULT_COLOR)
        rf_nodes.append({
            "id": nid,
            "type": "lineageNode",
            "data": {
                "operation":   node["operation"],
                "description": node["description"],
                "connector":   node.get("connector"),
                "source":      node.get("source"),
                "metadata":    node.get("metadata", {}),
                "timestamp":   node["timestamp"],
                "color":       color,
            },
            "position": {"x": depth * X_GAP, "y": lane * Y_GAP},
        })
        return nid

    def connect(src: str, dst: str) -> None:
        rf_edges.append({
            "id": f"e{src}-{dst}",
            "source": src,
            "target": dst,
            "type": "smoothstep",
        })

    def build(seq: list[dict], lane: int) -> tuple[str, int]:
        """Emit ``seq`` as a (possibly branching) chain; return (tip_id, tip_depth)."""
        tip = seq[-1]
        rcl = (tip.get("metadata") or {}).get("right_chain_len")
        is_branching_join = (
            tip["operation"].lower() == "join"
            and isinstance(rcl, int)
            and 0 < rcl < len(seq)
        )
        if is_branching_join:
            left_seq = seq[: len(seq) - 1 - rcl]
            right_seq = seq[len(seq) - 1 - rcl: -1]
            left_tip, left_depth = (build(left_seq, lane)
                                    if left_seq else (None, -1))
            lanes["max"] += 1
            right_tip, right_depth = build(right_seq, lanes["max"])
            depth = max(left_depth, right_depth) + 1
            nid = emit(tip, depth, lane)
            if left_tip is not None:
                connect(left_tip, nid)
            connect(right_tip, nid)
            return nid, depth

        prev_tip, prev_depth = (build(seq[:-1], lane)
                                if len(seq) > 1 else (None, -1))
        depth = prev_depth + 1
        nid = emit(tip, depth, lane)
        if prev_tip is not None:
            connect(prev_tip, nid)
        return nid, depth

    if nodes:
        build(list(nodes), 0)
    return {"nodes": rf_nodes, "edges": rf_edges}
