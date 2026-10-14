"""Derive one cited happy-path sequence per entry point from a block diagram."""

from __future__ import annotations

from collections import deque

from rox_dox.model import BlockDiagram, Edge, Participant, Sequence, Step

_UNGROUPED_COLUMN = 1_000_000


def _column_order(diagram: BlockDiagram) -> dict[str, int]:
    """Column index of every node and group; nodes outside a group sort last."""
    group_column = {group.id: index for index, group in enumerate(diagram.groups)}
    order = dict(group_column)
    for node in diagram.nodes:
        order[node.id] = (
            group_column[node.group] if node.group is not None else _UNGROUPED_COLUMN
        )
    return order


def _participants(diagram: BlockDiagram) -> dict[str, Participant]:
    endpoints = {
        node.id: Participant(id=node.id, label=node.label, source=node.source)
        for node in diagram.nodes
    }
    endpoints.update(
        {
            group.id: Participant(id=group.id, label=group.label, source=group.source)
            for group in diagram.groups
        }
    )
    return endpoints


def _walk(
    root: str,
    outgoing: dict[str, list[Edge]],
) -> tuple[list[str], list[Step]]:
    """Breadth-first from `root`; every edge becomes one step, each node once."""
    visited = [root]
    seen = {root}
    steps: list[Step] = []
    pending = deque([root])
    while pending:
        current = pending.popleft()
        for edge in outgoing.get(current, []):
            steps.append(
                Step(
                    src=edge.src,
                    dst=edge.dst,
                    message=edge.label,
                    source=edge.source,
                )
            )
            if edge.dst not in seen:
                seen.add(edge.dst)
                visited.append(edge.dst)
                pending.append(edge.dst)
    return visited, steps


def block_sequences(diagram: BlockDiagram) -> list[Sequence]:
    """One sequence per block-diagram root (node with edges out and none in).

    Steps follow the block's left-to-right flow, so the result is the cited
    happy path a request or queue message takes through the feature.
    """
    column = _column_order(diagram)
    participants = _participants(diagram)
    outgoing: dict[str, list[Edge]] = {}
    for edge in sorted(diagram.edges, key=lambda e: (column[e.dst], e.dst)):
        outgoing.setdefault(edge.src, []).append(edge)
    incoming = {edge.dst for edge in diagram.edges}
    roots = sorted(
        (source for source in outgoing if source not in incoming),
        key=lambda source: (column[source], source),
    )
    sequences = []
    for root in roots:
        visited, steps = _walk(root, outgoing)
        ordered = sorted(
            visited, key=lambda node_id: (column[node_id], visited.index(node_id))
        )
        sequences.append(
            Sequence(
                title=f"Flow from {participants[root].label}",
                participants=[participants[node_id] for node_id in ordered],
                steps=steps,
            )
        )
    return sequences
