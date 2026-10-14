"""Stable finding identity and explanations for static observations.

Identity ignores line numbers and parser byte offsets. ``sink_occurrence`` is
the ordinal of identical callee and whitespace-insensitive argument shape in
the same scope. The key also compacts the evidence snippet by removing
whitespace, so formatting changes do not churn identity. Spaces inside
string literals are not distinguished by that compaction. A harmless
``eval("constant")`` does not shift ``eval(request.args.get("q"))``.
Inserting another identical finding before an existing one does change later
ordinals; removing it restores them. Renaming a callee or changing an
argument token does. Explanations use only metadata the observation already
recorded. Missing relationships are omitted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from app.security.correlation import ObservationCluster
from app.security.rules.base import SecurityObservation
from app.security.taint import field_path_from_reason


@dataclass(frozen=True)
class FlowExplanation:
    """A concise account of one static finding. Not a verification."""

    finding_key: str
    related_group: str
    summary: str
    source: str
    sink: str
    field_path: str
    files_crossed: str
    analysis_incomplete: str
    parser_completeness: str
    evidence_summary: str


def semantic_clusters(observations: list[SecurityObservation]) -> list[ObservationCluster]:
    """Group observations that describe the same sink, not merely the same file."""
    grouped: dict[tuple[str, ...], list[SecurityObservation]] = {}
    for obs in observations:
        grouped.setdefault(_identity_parts(obs), []).append(obs)
    clusters: list[ObservationCluster] = []
    for parts, items in grouped.items():
        items.sort(key=lambda obs: (obs.line, obs.rule_id, obs.evidence_text))
        clusters.append(
            ObservationCluster(
                vulnerability_class=items[0].vulnerability_class,
                file_path=items[0].file_path,
                observations=tuple(items),
            )
        )
    clusters.sort(
        key=lambda cluster: (cluster.file_path, cluster.line, cluster.vulnerability_class.value)
    )
    return clusters


def explain_cluster(
    observations: tuple[SecurityObservation, ...] | list[SecurityObservation],
) -> FlowExplanation:
    ordered = tuple(
        sorted(observations, key=lambda obs: (obs.line, obs.rule_id, obs.evidence_text))
    )
    if not ordered:
        return FlowExplanation("", "", "", "", "", "", "", "", "", "")
    key = _finding_key(ordered[0])
    group = _related_group(ordered[0])
    source = _first(ordered, _source_label)
    sink = _first(ordered, _sink_label)
    field_path = _first(ordered, _field_label)
    files = _files(ordered)
    incomplete = _first(ordered, lambda obs: str(obs.metadata.get("analysis_incomplete") or ""))
    parser = _parser_text(ordered)
    steps = _steps(ordered, source, sink, field_path)
    evidence = _evidence_text(ordered)
    return FlowExplanation(
        finding_key=key,
        related_group=group,
        summary=" → ".join(steps),
        source=source,
        sink=sink,
        field_path=field_path,
        files_crossed=",".join(files),
        analysis_incomplete=incomplete,
        parser_completeness=parser,
        evidence_summary=evidence,
    )


def _identity_parts(obs: SecurityObservation) -> tuple[str, ...]:
    return (
        obs.vulnerability_class.value,
        obs.file_path.replace("\\", "/"),
        obs.scope_id,
        _sink_label(obs),
        _field_label(obs),
        str(obs.metadata.get("argument_index") or obs.argument_index or ""),
        _source_family(obs),
        "".join(obs.evidence_text.split()),
        str(obs.metadata.get("sink_occurrence") or ""),
    )


def _finding_key(obs: SecurityObservation) -> str:
    return _digest(_identity_parts(obs))


def _related_group(obs: SecurityObservation) -> str:
    return _digest(
        (
            obs.vulnerability_class.value,
            _sink_label(obs),
            _field_label(obs),
            _source_family(obs),
        )
    )


def _digest(parts: tuple[str, ...]) -> str:
    payload = json.dumps(parts, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _source_family(obs: SecurityObservation) -> str:
    taint = str(obs.metadata.get("taint") or obs.taint_path or "")
    marker = "source:"
    index = taint.find(marker)
    if index < 0:
        return ""
    token = taint[index + len(marker) :].split("|", 1)[0].split(":", 1)[0].strip()
    if not token:
        return ""
    return f"source:{token}"


def _source_label(obs: SecurityObservation) -> str:
    family = _source_family(obs)
    if family:
        return family.removeprefix("source:")
    explicit = str(obs.metadata.get("taint_source") or "")
    if explicit.startswith("source:"):
        explicit = explicit.removeprefix("source:")
    return explicit.split(":", 1)[0] if explicit else ""


def _sink_label(obs: SecurityObservation) -> str:
    recorded = str(obs.metadata.get("callee_sink") or "").strip()
    if recorded:
        return recorded
    return str(obs.metadata.get("sink") or obs.sink_id or "").strip()


def _field_label(obs: SecurityObservation) -> str:
    recorded = str(obs.metadata.get("field_path") or "")
    if recorded:
        return recorded
    return field_path_from_reason(str(obs.metadata.get("taint") or obs.taint_path or ""))


def _first(
    observations: tuple[SecurityObservation, ...],
    read: Callable[[SecurityObservation], str],
) -> str:
    values: list[str] = []
    for obs in observations:
        value = read(obs).strip()
        if value and value not in values:
            values.append(value)
    return values[0] if values else ""


def _files(observations: tuple[SecurityObservation, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for obs in observations:
        for key in ("file_path",):
            _add_path(found, getattr(obs, key))
        for key in ("callee_file", "defining_file", "caller_file"):
            _add_path(found, str(obs.metadata.get(key) or ""))
    return tuple(sorted(found))


def _add_path(found: list[str], raw: str) -> None:
    path = raw.replace("\\", "/").strip()
    if path and path not in found:
        found.append(path)


def _steps(
    observations: tuple[SecurityObservation, ...],
    source: str,
    sink: str,
    field_path: str,
) -> list[str]:
    steps: list[str] = []
    if source:
        steps.append(f"source {source}")
    if field_path:
        steps.append(f"field {field_path}")
    relationship = _meta_value(observations, "relationship")
    if relationship:
        steps.append(relationship)
    if _meta_value(observations, "re_export") == "true" and relationship != "re_export":
        steps.append("re_export")
    callee_file = _meta_value(observations, "callee_file")
    callee_function = _meta_value(observations, "callee_function")
    caller = observations[0].file_path.replace("\\", "/")
    if callee_file and callee_function and callee_file.replace("\\", "/") != caller:
        steps.append(f"helper {callee_file}::{callee_function}")
    elif callee_function and relationship == "alias":
        steps.append(f"helper {callee_function}")
    if sink:
        steps.append(f"sink {sink}")
    return steps


def _raw_meta(observations: tuple[SecurityObservation, ...], key: str) -> str:
    for obs in observations:
        if key in obs.metadata:
            return str(obs.metadata.get(key) or "")
    return ""


def _meta_value(observations: tuple[SecurityObservation, ...], key: str) -> str:
    for obs in observations:
        value = str(obs.metadata.get(key) or "").strip()
        if value and value not in {"false", "none"}:
            return value
    return ""


def _parser_text(observations: tuple[SecurityObservation, ...]) -> str:
    parts: list[str] = []
    backend = _meta_value(observations, "parser_backend") or observations[0].parser_backend
    tier = _meta_value(observations, "parser_tier") or observations[0].parser_tier
    if backend or tier:
        parts.append(f"parser {backend} {tier}".strip())
    if "profile_fallback" in tier:
        parts.append("profile fallback is not dataflow")
    if (
        _raw_meta(observations, "parser_complete") == "false"
        or _raw_meta(observations, "cross_file_partial") == "true"
    ):
        parts.append("callee parse is partial")
    incomplete = _meta_value(observations, "analysis_incomplete")
    if incomplete:
        parts.append(f"incomplete: {incomplete}")
    return "; ".join(parts)


def _evidence_text(observations: tuple[SecurityObservation, ...]) -> str:
    rules = sorted({obs.rule_id for obs in observations})
    confidence = sorted({obs.confidence for obs in observations})[-1] if observations else ""
    kind = observations[0].vulnerability_class.value
    return (
        f"static analysis of {kind} via {', '.join(rules)}; confidence {confidence}; not verified"
    )
