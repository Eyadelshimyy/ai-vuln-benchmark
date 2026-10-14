"""Typed writers for the VibeBB Record Protocol (VRP) v2.

Every sibling plugin keeps the same append-only logs under
``observations/<plugin>/`` (see ``docs/records-protocol.md``). The rules
live in ``_vrp.py``, a byte-identical copy of the stdlib hook module
``plugins/fpga/hooks/scripts/_records.py``: the Pydantic models below give
the MCP tools their input schemas, and every record is re-checked with the
same functions the Stop hook uses before it is appended. Each line links
to the previous one by ``prev_event_id`` and carries the sha256 of its own
canonical content as ``event_id``.

Write-time checks need the workspace and are therefore only done here:
claims must quote tokens that occur in the cited artifact, a new
impression must not be a near copy of an earlier one, upstream citations
must resolve to intact sister logs, a concern another plugin disputed
cannot come back while the artifacts are unchanged, a rejected insight
cannot be proposed again without ``revisits``, and review rounds are
capped. Records are advisory evidence and never change a gate verdict.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import _vrp
from .workspace import workspace_path, workspace_root

PLUGIN: Final = "fpga"
SCHEMA_VERSION: Final = _vrp.SCHEMA_VERSION
RECORDS_DIR: Final = Path(_vrp.OBSERVATIONS_DIR) / PLUGIN
IMPRESSION_MIN_CHARS: Final = _vrp.IMPRESSION_MIN_CHARS
IMPRESSION_MIN_SENTENCES: Final = _vrp.IMPRESSION_MIN_SENTENCES
RATIONALE_MIN_CHARS: Final = _vrp.RATIONALE_MIN_CHARS
PRINCIPLE_MIN_CHARS: Final = _vrp.PRINCIPLE_MIN_CHARS
QUESTION_MIN_CHARS: Final = _vrp.QUESTION_MIN_CHARS
LOG_FILES: Final[dict[str, str]] = dict(_vrp.LOG_FILES)
_SLUG: Final = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_SHA256: Final = r"^[0-9a-f]{64}$"

sentence_count = _vrp.sentence_count
sha256_file = _vrp.sha256_file
tree_sha256 = _vrp.tree_sha256


def impression_is_prose(value: str) -> str:
    """Reject a terse status line where a long-form impression is required."""
    errors = _vrp.impression_errors(value)
    if errors:
        raise ValueError("; ".join(errors))
    return value.strip()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Severity = Literal["info", "warning", "error"]
Disposition = Literal["adopted", "deferred", "disputed", "noted"]


class ArtifactRef(_Strict):
    path: str = Field(min_length=1, description="Workspace-relative path")
    sha256: str = Field(pattern=_SHA256)


class EvidenceInput(_Strict):
    path: str | None = Field(default=None, description="Workspace-relative file (hashed)")
    reference: str | None = Field(
        default=None, description="Standard, datasheet or law cited when there is no file"
    )

    @model_validator(mode="after")
    def _one_kind(self) -> EvidenceInput:
        if self.path is None and not (self.reference and self.reference.strip()):
            raise ValueError("evidence needs a path (bound by sha256) or a reference")
        return self


class EventRef(_Strict):
    system: str = Field(pattern=_SLUG, description="Plugin that wrote the record")
    event_id: str = Field(pattern=_SHA256)


class InsightRef(EventRef):
    insight_id: str = Field(pattern=_SLUG)


class Concern(_Strict):
    id: str = Field(pattern=_SLUG)
    text: str = Field(min_length=_vrp.FACET_MIN_CHARS)
    severity: Severity
    about: str = Field(pattern=_SLUG, description="Plugin that owns it, or 'self'")
    anchor: str | None = Field(default=None, min_length=_vrp.ANCHOR_MIN_CHARS)


class Feelings(_Strict):
    maker: str = Field(min_length=_vrp.FACET_MIN_CHARS)
    user: str = Field(min_length=_vrp.FACET_MIN_CHARS)


class Facets(_Strict):
    observed: list[str] = Field(min_length=2)
    works: list[str] = Field(min_length=1)
    concerns: list[Concern] = Field(default_factory=list[Concern])
    no_concerns_reason: str | None = None
    feelings: Feelings
    next_actions: list[str] = Field(min_length=1)


class Claim(_Strict):
    text: str = Field(min_length=_vrp.FACET_MIN_CHARS)
    anchor: str = Field(
        min_length=_vrp.ANCHOR_MIN_CHARS,
        description="Exact token from the artifact (part number, net, cavity, label)",
    )
    artifact: str = Field(min_length=1, description="Workspace-relative file the claim reads")


class Insight(_Strict):
    id: str = Field(pattern=_SLUG)
    hypothesis: str = Field(min_length=_vrp.HYPOTHESIS_MIN_CHARS)
    proposed_change: str
    expected_effect: str
    test: str = Field(description="How to check it, ideally a deterministic gate")
    target: str = Field(pattern=_SLUG, description="Plugin that would change")
    revisits: InsightRef | None = None
    revisit_reason: str | None = None


class Upstream(_Strict):
    system: str = Field(pattern=_SLUG)
    event_id: str = Field(pattern=_SHA256)
    disposition: Disposition
    effect: str = Field(min_length=_vrp.EFFECT_MIN_CHARS)
    concern_ids: list[str] | None = None


class _ImpressionBody(_Strict):
    impression: str
    facets: Facets
    claims: list[Claim]
    insights: list[Insight] = Field(default_factory=list[Insight])
    upstream: list[Upstream] = Field(default_factory=list[Upstream])
    confidence: Literal["low", "medium", "high"]
    unknowns: list[str] = Field(default_factory=list[str])
    delta: str | None = Field(
        default=None, description="What is new when an earlier impression reads similarly"
    )

    _check_impression = field_validator("impression")(impression_is_prose)


class StageImpressionInput(_ImpressionBody):
    stage: str = Field(pattern=_SLUG)
    artifacts: list[str] = Field(
        min_length=1, description="Workspace-relative files or directories the stage produced"
    )


class VisionFinding(_Strict):
    category: str = Field(min_length=1)
    severity: Severity
    note: str = Field(min_length=1)


class Lookback(_Strict):
    claim: int = Field(ge=0, description="Index into claims")
    verdict: Literal["confirmed", "refuted", "uncertain"]
    note: str = Field(min_length=_vrp.FACET_MIN_CHARS)


class VisionReviewInput(_ImpressionBody):
    image_path: str | None = Field(default=None, description="Workspace-relative image")
    source_event_id: str | None = Field(
        default=None, description="event_id of the vision tool event or image observation"
    )
    model: str = Field(min_length=1)
    checklist: str = Field(pattern=_SLUG)
    reviewer: Literal["primary", "blind", "tiebreak"] = "primary"
    findings: list[VisionFinding] = Field(default_factory=list[VisionFinding])
    lookback: list[Lookback]

    @model_validator(mode="after")
    def _bound(self) -> VisionReviewInput:
        if self.image_path is None and self.source_event_id is None:
            raise ValueError("a vision review needs image_path or source_event_id")
        if self.reviewer != "primary" and self.image_path is None:
            raise ValueError("blind and tiebreak reviews must bind image_path")
        return self


class DecisionOption(_Strict):
    name: str = Field(min_length=1)
    pros: list[str] = Field(min_length=1)
    cons: list[str] = Field(min_length=1)


class DecisionInput(_Strict):
    id: str = Field(pattern=_SLUG, description="Stable decision id, e.g. 'fpga-clock-plan'")
    stage: str = Field(pattern=_SLUG)
    question: str = Field(min_length=QUESTION_MIN_CHARS)
    principles: list[str] = Field(
        min_length=1,
        description="First principles, physical laws or standards the choice rests on",
    )
    options: list[DecisionOption] = Field(min_length=2)
    chosen: str = Field(min_length=1)
    rationale: str = Field(min_length=RATIONALE_MIN_CHARS)
    evidence: list[EvidenceInput] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list[str])
    unknowns: list[str] = Field(default_factory=list[str])
    risks: list[str] = Field(min_length=1)
    revisit_when: str = Field(min_length=1)
    decided_by: Literal["agent", "user"] = "agent"
    impression_refs: list[EventRef] = Field(
        default_factory=list[EventRef], description="Impressions (any plugin) behind it"
    )
    insight_refs: list[InsightRef] = Field(default_factory=list[InsightRef])

    @field_validator("principles")
    @classmethod
    def _principles_are_cited(cls, value: list[str]) -> list[str]:
        if any(len(item.strip()) < PRINCIPLE_MIN_CHARS for item in value):
            raise ValueError(f"each principle needs at least {PRINCIPLE_MIN_CHARS} characters")
        return value

    @model_validator(mode="after")
    def _chosen_is_an_option(self) -> DecisionInput:
        names = [option.name for option in self.options]
        if len(set(names)) != len(names):
            raise ValueError("option names must be unique")
        if self.chosen not in names:
            raise ValueError("chosen must name one of the options")
        return self


class GateResult(_Strict):
    gate: str = Field(min_length=1)
    verdict: Literal["pass", "fail", "unknown"]


class InsightStatusInput(_Strict):
    insight: InsightRef
    status: Literal["tried", "adopted", "rejected", "deferred", "superseded"]
    reason: str = Field(min_length=_vrp.REASON_MIN_CHARS)
    evidence: list[EvidenceInput] = Field(min_length=1)
    gate_verdicts: list[GateResult] = Field(default_factory=list[GateResult])
    decision_refs: list[str] = Field(default_factory=list[str])
    revisit_when: str | None = None


class ReconcileInput(_Strict):
    reviews: list[str] = Field(
        min_length=2,
        max_length=3,
        description="event_ids of [primary, blind] or [primary, blind, tiebreak]",
    )


class SongReceiptInput(_Strict):
    delivery: str = Field(description="liaison/<id>.bard-song.json addressed to this plugin")
    felt: str = Field(description="What the song made you feel about the design (160+ chars)")
    prompted_review: bool
    follow_up: str | None = None


class ReadInput(_Strict):
    artifact: str = Field(description="Workspace-relative sister artifact you took in")
    producer: str = Field(pattern=_SLUG, description="Plugin that produced it")


def records_dir(root: Path | None = None) -> Path:
    return (root or workspace_root()) / RECORDS_DIR


def _relative(value: str, root: Path) -> tuple[str, Path]:
    path = workspace_path(value, root)
    if not path.exists():
        raise ValueError(f"artifact does not exist: {value}")
    relative = path.relative_to(root.resolve()).as_posix() if path != root.resolve() else "."
    return relative, path


def _archive_legacy(log: Path) -> None:
    records, _ = _vrp.load_jsonl(log)
    if not _vrp.is_legacy(records):
        return
    index = 1
    target = log.with_name(f"{log.stem}.v1.jsonl")
    while target.exists():
        index += 1
        target = log.with_name(f"{log.stem}.v1-{index}.jsonl")
    log.rename(target)


def _append(kind: str, body: dict[str, Any], root: Path) -> dict[str, Any]:
    log = records_dir(root) / LOG_FILES[kind]
    log.parent.mkdir(parents=True, exist_ok=True)
    with (log.parent / ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _archive_legacy(log)
        existing, problems = _vrp.verify_log(log, kind)
        if problems:
            raise ValueError(
                f"{log.name} fails integrity, refusing to append: {problems[0]}. Move the"
                " file aside (it stays as evidence) to start a new chain."
            )
        now = datetime.now(UTC)
        last = _vrp.parse_time(existing[-1].get("recorded_at")) if existing else None
        if last is not None and now.timestamp() < last:
            raise ValueError("system clock is behind the last record; refusing to append")
        writer: dict[str, Any] = {"plugin": PLUGIN}
        for key, env in (("agent", "VRP_AGENT"), ("model", "VRP_MODEL")):
            if os.environ.get(env, "").strip():
                writer[key] = os.environ[env].strip()
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "plugin": PLUGIN,
            "sequence": len(existing) + 1,
            "prev_event_id": _vrp.chain_head(existing),
            "recorded_at": now.isoformat(),
            "writer": writer,
            **{key: value for key, value in body.items() if value is not None},
        }
        record["event_id"] = _vrp.compute_event_id(record)
        errors = _vrp.record_errors(kind, record)
        if errors:
            raise ValueError(f"invalid {kind}: " + "; ".join(errors))
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"verdict": "pass", "kind": kind, "path": str(log), "record": record}


def _workspace(base: Path) -> dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]]:
    return _vrp.plugin_logs(base)


def _own(
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
) -> dict[str, list[dict[str, Any]]]:
    logs, problems = workspace.get(PLUGIN, ({}, []))
    if problems:
        raise ValueError(f"{PLUGIN} records fail integrity: {problems[0]}")
    return logs


def _context_errors(
    kind: str, body: dict[str, Any], base: Path, artifacts: list[dict[str, str]]
) -> list[str]:
    workspace = _workspace(base)
    own = _own(workspace)
    draft = {"kind": kind, **body, "artifacts": artifacts} if artifacts else {"kind": kind, **body}
    errors = _vrp.grounding_errors(draft, base)
    errors += _vrp.novelty_errors(draft, own.get(kind, []))
    errors += _vrp.upstream_resolution_errors(draft, workspace)
    errors += _vrp.reraise_errors(draft, PLUGIN, workspace)
    errors += _vrp.rejected_insight_errors(draft, workspace)
    return errors


def _hash_evidence(items: list[EvidenceInput], base: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in items:
        if item.path is not None:
            relative, path = _relative(item.path, base)
            evidence.append({"path": relative, "sha256": tree_sha256(path)})
        else:
            evidence.append({"reference": item.reference})
    return evidence


def _dump(model: BaseModel, exclude: set[str] | None = None) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True, exclude=exclude)


def record_decision(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    base = (root or workspace_root()).resolve()
    decision = DecisionInput.model_validate(dict(payload))
    body = _dump(decision, {"evidence"})
    body["evidence"] = _hash_evidence(list(decision.evidence), base)
    workspace = _workspace(base)
    problems = [
        p
        for ref in decision.impression_refs
        if (p := _vrp.resolve_ref(workspace, ref.system, ref.event_id)[1])
    ]
    for ref in decision.insight_refs:
        target, problem = _vrp.resolve_ref(workspace, ref.system, ref.event_id)
        ids = {
            str(i.get("id"))
            for i in cast(list[dict[str, Any]], (target or {}).get("insights") or [])
        }
        if problem or ref.insight_id not in ids:
            problems.append(problem or f"insight {ref.insight_id} is not in that impression")
    if problems:
        raise ValueError("unresolved refs: " + "; ".join(problems))
    return _append("decision", body, base)


def record_impression(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    base = (root or workspace_root()).resolve()
    impression = StageImpressionInput.model_validate(dict(payload))
    artifacts: list[dict[str, str]] = []
    for value in impression.artifacts:
        relative, path = _relative(value, base)
        artifacts.append({"path": relative, "sha256": tree_sha256(path)})
    body = _dump(impression, {"artifacts"})
    errors = _context_errors("stage_impression", body, base, artifacts)
    if errors:
        raise ValueError("; ".join(errors))
    body["artifacts"] = artifacts
    return _append("stage_impression", body, base)


def record_vision_review(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    base = (root or workspace_root()).resolve()
    review = VisionReviewInput.model_validate(dict(payload))
    body = _dump(review)
    if review.image_path is not None:
        relative, path = _relative(review.image_path, base)
        if not path.is_file():
            raise ValueError(f"image is not a file: {review.image_path}")
        body["image_path"] = relative
        body["image_sha256"] = sha256_file(path)
    own = _own(_workspace(base))
    errors = _context_errors("vision_review", body, base, [])
    errors += _round_errors(review.reviewer, body.get("image_sha256"), own)
    if errors:
        raise ValueError("; ".join(errors))
    return _append("vision_review", body, base)


def _round_errors(reviewer: str, digest: Any, own: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Blind needs a primary; a tiebreak needs a round-1 split; each happens once."""
    if reviewer == "primary":
        return []
    reviews = [r for r in own.get("vision_review", []) if r.get("image_sha256") == digest]
    primaries = [r for r in reviews if r.get("reviewer") == "primary"]
    if not primaries:
        return [f"a {reviewer} review needs a primary review of the same image bytes first"]
    primary = primaries[-1]
    after = _vrp.parse_time(primary.get("recorded_at")) or 0.0
    later = [
        r
        for r in reviews
        if r.get("reviewer") == reviewer and (_vrp.parse_time(r.get("recorded_at")) or 0.0) >= after
    ]
    if later:
        return [f"this image already has its {reviewer} review; reviews are not repeated"]
    if reviewer == "tiebreak":
        splits = [
            c
            for c in own.get("review_reconcile", [])
            if primary.get("event_id") in cast(list[str], c.get("reviews") or [])
            and c.get("outcome") == "disagree"
        ]
        if not splits:
            return ["a tiebreak review is only allowed after a round-1 reconcile that disagrees"]
    return []


def record_reconcile(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    base = (root or workspace_root()).resolve()
    request = ReconcileInput.model_validate(dict(payload))
    own = _own(_workspace(base))
    by_id = {str(r.get("event_id")): r for r in own.get("vision_review", [])}
    missing = [event for event in request.reviews if event not in by_id]
    if missing:
        raise ValueError(f"unknown vision review event ids: {missing}")
    reviews = [by_id[event] for event in request.reviews]
    roles = [str(r.get("reviewer")) for r in reviews]
    expected = ["primary", "blind", "tiebreak"][: len(reviews)]
    if roles != expected:
        raise ValueError(f"reviews must be ordered {expected}, got {roles}")
    digests = {r.get("image_sha256") for r in reviews}
    if len(digests) != 1 or None in digests:
        raise ValueError("reconciled reviews must bind the same image bytes")
    prior = [
        c
        for c in own.get("review_reconcile", [])
        if request.reviews[0] in cast(list[str], c.get("reviews") or [])
    ]
    result = _vrp.reconcile(reviews[0], reviews[1:])
    if any(int(c.get("round") or 0) >= result["round"] for c in prior):
        raise ValueError(f"round {result['round']} is already reconciled for this primary review")
    if result["round"] == 2 and not any(c.get("outcome") == "disagree" for c in prior):
        raise ValueError("round 2 needs a round-1 reconcile that disagrees")
    body = {"image_sha256": digests.pop(), "reviews": list(request.reviews), **result}
    return _append("review_reconcile", body, base)


def record_insight(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    base = (root or workspace_root()).resolve()
    status = InsightStatusInput.model_validate(dict(payload))
    body = _dump(status, {"evidence"})
    body["evidence"] = _hash_evidence(list(status.evidence), base)
    errors = _vrp.insight_write_errors(body, _workspace(base))
    if errors:
        raise ValueError("; ".join(errors))
    return _append("insight_status", body, base)


def record_song_receipt(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Receive a bard song; never asks bard for another one (no record-calls-record loop)."""
    base = (root or workspace_root()).resolve()
    receipt = SongReceiptInput.model_validate(dict(payload))
    relative, path = _relative(receipt.delivery, base)
    if not relative.startswith(_vrp.LIAISON_DIR + "/") or not relative.endswith(
        _vrp.SONG_DELIVERY_SUFFIX
    ):
        raise ValueError(f"delivery must be liaison/<id>{_vrp.SONG_DELIVERY_SUFFIX}")
    doc: Any = json.loads(path.read_text(encoding="utf-8"))
    problems = _vrp.song_delivery_errors(doc)
    if problems:
        raise ValueError("song delivery is malformed: " + "; ".join(problems))
    song = cast(dict[str, Any], doc)
    if song.get("to") != PLUGIN:
        raise ValueError(f"this song is addressed to {song.get('to')}, not {PLUGIN}")
    digest = sha256_file(path)
    if _vrp.received(_own(_workspace(base)), relative, digest):
        raise ValueError("this song delivery already has a receipt")
    body = _dump(receipt, {"delivery"})
    body |= {"delivery": {"path": relative, "sha256": digest}, "song_id": song["id"]}
    return _append("song_receipt", body, base)


def impressions_for_artifact(
    path: Path, producer: str, root: Path | None = None
) -> tuple[list[dict[str, str]], list[str]]:
    """(refs, unresolved) for a sister artifact: its explicit impression_refs plus
    every intact producer impression bound to exactly these bytes."""
    base = (root or workspace_root()).resolve()
    workspace = _workspace(base)
    refs: list[dict[str, str]] = []
    unresolved: list[str] = []
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        value = None
    explicit: list[Any] = []
    if isinstance(value, dict):
        doc = cast(dict[str, Any], value)
        provenance = doc.get("provenance")
        explicit = cast(list[Any], doc.get("impression_refs") or [])
        if isinstance(provenance, dict):
            explicit += cast(
                list[Any], cast(dict[str, Any], provenance).get("impression_refs") or []
            )
    for item in explicit:
        if not isinstance(item, dict):
            unresolved.append("malformed impression_ref")
            continue
        entry = cast(dict[str, Any], item)
        system, event = str(entry.get("system") or producer), str(entry.get("event_id"))
        _, problem = _vrp.resolve_ref(workspace, system, event)
        if problem:
            unresolved.append(problem)
        elif {"system": system, "event_id": event} not in refs:
            refs.append({"system": system, "event_id": event})
    logs, problems = workspace.get(producer, ({}, []))
    if problems:
        unresolved.append(f"{producer} records fail integrity: {problems[0]}")
    else:
        digest = sha256_file(path)
        for kind in _vrp.IMPRESSION_KINDS:
            for record in logs.get(kind, []):
                shas = [
                    str(a.get("sha256"))
                    for a in cast(list[dict[str, Any]], record.get("artifacts") or [])
                ]
                if digest in shas or record.get("image_sha256") == digest:
                    ref = {"system": producer, "event_id": str(record.get("event_id"))}
                    if ref not in refs:
                        refs.append(ref)
    if not refs and not unresolved:
        unresolved.append(f"no {producer} impression is bound to {path.name}")
    return refs, unresolved


def record_read(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Record that a sister artifact was taken in, with the impressions that came with it."""
    base = (root or workspace_root()).resolve()
    read = ReadInput.model_validate(dict(payload))
    relative, path = _relative(read.artifact, base)
    if not path.is_file():
        raise ValueError(f"artifact is not a file: {read.artifact}")
    refs, unresolved = impressions_for_artifact(path, read.producer, base)
    body = {
        "artifact": {"path": relative, "sha256": sha256_file(path)},
        "producer": read.producer,
        "refs": refs,
        "unresolved": unresolved,
    }
    return _append("upstream_read", body, base)


RECORDERS = {
    "decision": record_decision,
    "impression": record_impression,
    "vision-review": record_vision_review,
    "reconcile": record_reconcile,
    "insight": record_insight,
    "song-receipt": record_song_receipt,
    "read": record_read,
}


def records_summary(root: Path | None = None) -> dict[str, Any]:
    """Counts per log, chain heads, integrity and the last Stop-hook verdict."""
    directory = records_dir(root)
    counts: dict[str, int] = {}
    heads: dict[str, str] = {}
    integrity: list[str] = []
    for kind, filename in LOG_FILES.items():
        records, problems = _vrp.verify_log(directory / filename, kind)
        counts[kind] = len(records)
        heads[kind] = _vrp.chain_head(records)
        integrity += problems
    status_path = directory / "records-status.json"
    status: Any = None
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {"verdict": "fail", "problems": ["records-status.json is malformed"]}
    return {
        "verdict": "unknown" if integrity else "pass",
        "records_dir": str(directory),
        "counts": counts,
        "heads": heads,
        "integrity_problems": integrity,
        "last_stop": status,
    }


def records_digest(root: Path | None = None) -> dict[str, Any]:
    return _vrp.digest((root or workspace_root()).resolve())


def records_search(payload: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    allowed = {"query", "system", "kind", "stage", "artifact", "severity", "open_only", "limit"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown search fields: {unknown}")
    return _vrp.search((root or workspace_root()).resolve(), **dict(payload))


def _event_ids(log: Path, kind: str) -> set[str]:
    """Event ids from an intact VRP v2 log; a broken chain refuses every ref."""
    records, problems = _vrp.verify_log(log, kind)
    if problems:
        raise ValueError(f"VRP {kind} log fails integrity: {problems[0]}")
    return {str(record.get("event_id")) for record in records}


def decision_event_ids(root: Path | None = None) -> set[str]:
    return _event_ids(records_dir(root) / "decisions.jsonl", "decision")


def impression_event_ids(root: Path | None = None) -> set[str]:
    directory = records_dir(root)
    return _event_ids(directory / "impressions.jsonl", "stage_impression") | _event_ids(
        directory / "vision-reviews.jsonl", "vision_review"
    )
