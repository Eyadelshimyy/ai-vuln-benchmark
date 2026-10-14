"""VibeBB Record Protocol (VRP) v2 — stdlib rules shared by every plugin.

Append-only JSONL logs live under ``<records_dir>`` (from the plugin's
``hooks/records-policy.json``, normally ``observations/<plugin>``):

* ``decisions.jsonl`` — design decisions with first principles, at least
  two options, evidence bound by sha256, unknowns, risks and a revisit
  trigger, optionally citing the impressions and insights behind them;
* ``impressions.jsonl`` — one structured long-form impression per
  finished stage, bound to the artifacts it read and grounded in them;
* ``vision-reviews.jsonl`` — what a reviewer thought after looking at an
  image (primary, blind or tiebreak), with a look-back per claim;
* ``reads.jsonl`` — sister impressions this plugin took in with an
  imported artifact (each must be answered by a later impression);
* ``insights.jsonl`` — lifecycle of the hypotheses impressions propose;
* ``reconciles.jsonl`` — deterministic comparison of independent reviews;
* ``song-receipts.jsonl`` — how this plugin received a bard song.

Every line carries ``prev_event_id`` and an ``event_id`` that is the
sha256 of the canonical record, so deleting, reordering or editing a line
breaks the chain. The chain shows tampering; it does not prove who wrote
a line. The core writers import this module (vendored as
``<package>/_vrp.py``) and hooks re-validate with it, so one rule set
serves both. Records are advisory evidence: they never change a gate
verdict and only ever push toward more work.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = 2
IMPRESSION_MIN_CHARS = 400
IMPRESSION_MIN_SENTENCES = 3
RATIONALE_MIN_CHARS = 200
PRINCIPLE_MIN_CHARS = 12
QUESTION_MIN_CHARS = 10
FACET_MIN_CHARS = 8
EFFECT_MIN_CHARS = 40
REASON_MIN_CHARS = 40
HYPOTHESIS_MIN_CHARS = 20
SONG_FEELING_MIN_CHARS = 160
SONG_FEELING_MIN_SENTENCES = 2
ANCHOR_MIN_CHARS = 2
STAGE_CLAIMS_MIN = 2
NEAR_DUPLICATE_REJECT = 0.80
NEAR_DUPLICATE_DELTA = 0.60
RERAISE_SIMILARITY = 0.60
SHINGLE_SIZE = 5
RECONCILE_AGREE_JACCARD = 0.5
MAX_REVIEW_ROUNDS = 2
GROUNDING_MAX_BYTES = 4_000_000
GROUNDING_MAX_FILES = 2000
DEFAULT_MAX_STOP_DENIALS = 2
GENESIS = "0" * 64
LOG_FILES = {
    "decision": "decisions.jsonl",
    "stage_impression": "impressions.jsonl",
    "vision_review": "vision-reviews.jsonl",
    "upstream_read": "reads.jsonl",
    "insight_status": "insights.jsonl",
    "review_reconcile": "reconciles.jsonl",
    "song_receipt": "song-receipts.jsonl",
}
IMPRESSION_KINDS = ("stage_impression", "vision_review")
VISION_EVENTS_FILE = "vision-tool-events.jsonl"
IMAGE_OBSERVATIONS_FILE = "image-observations.jsonl"
SONG_DELIVERY_SUFFIX = ".bard-song.json"
LIAISON_DIR = "liaison"
OBSERVATIONS_DIR = "observations"
SKIP_PARTS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
SEVERITIES = {"info", "warning", "error"}
DECIDERS = {"agent", "user"}
CONFIDENCES = {"low", "medium", "high"}
DISPOSITIONS = {"adopted", "deferred", "disputed", "noted"}
REVIEWERS = {"primary", "blind", "tiebreak"}
LOOKBACK_VERDICTS = {"confirmed", "refuted", "uncertain"}
INSIGHT_STATUSES = {"tried", "adopted", "rejected", "deferred", "superseded"}
INSIGHT_TRANSITIONS = {
    "proposed": {"tried", "rejected", "deferred"},
    "tried": {"tried", "adopted", "rejected", "deferred"},
    "deferred": {"tried", "rejected"},
    "adopted": {"superseded"},
    "rejected": set[str](),
    "superseded": set[str](),
}
GATE_VERDICTS = {"pass", "fail", "unknown"}
RECONCILE_OUTCOMES = {"agree", "disagree", "unresolved"}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CJK_END = re.compile(r"[。\uff01\uff1f]")
_LATIN_END = re.compile(r"[.!?](?=\s|$)")
_SPACE = re.compile(r"\s+")


# --- text rules -----------------------------------------------------------


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。\uff01\uff1f])|(?<=[.!?])(?=\s|$)", text)
    return [part.strip() for part in parts if part.strip()]


def sentence_count(text: str) -> int:
    return len(_CJK_END.findall(text)) + len(_LATIN_END.findall(text))


def prose_errors(
    value: Any,
    field: str = "impression",
    min_chars: int = IMPRESSION_MIN_CHARS,
    min_sentences: int = IMPRESSION_MIN_SENTENCES,
) -> list[str]:
    if not isinstance(value, str):
        return [f"{field} must be a string"]
    text = value.strip()
    errors: list[str] = []
    if len(text) < min_chars:
        errors.append(
            f"{field} has {len(text)} characters; write at least {min_chars} (what you"
            " noticed, what works, what worries you, what a maker or user would feel,"
            " what to do next)"
        )
    count = sentence_count(text)
    if count < min_sentences:
        errors.append(f"{field} must contain at least {min_sentences} sentences")
    if count >= min_sentences and len(set(sentences(text))) < min_sentences:
        errors.append(f"{field} repeats the same sentence")
    return errors


def impression_errors(value: Any, field: str = "impression") -> list[str]:
    return prose_errors(value, field)


def normalized(text: str) -> str:
    return _SPACE.sub(" ", text.casefold()).strip()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-grams of the case-folded text; works for CJK and Latin alike."""
    flat = normalized(text)
    if len(flat) <= size:
        return {flat} if flat else set()
    return {flat[i : i + size] for i in range(len(flat) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def max_similarity(text: str, others: Iterable[str]) -> float:
    mine = shingles(text)
    return max((jaccard(mine, shingles(other)) for other in others), default=0.0)


# --- hashing --------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str:
    """sha256 of a file, or of a directory's sorted (relative path, sha256) list."""
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink()):
        relative = child.relative_to(path).as_posix()
        if any(part in SKIP_PARTS for part in child.relative_to(path).parts):
            continue
        digest.update(f"{relative}\0{sha256_file(child)}\n".encode())
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def compute_event_id(record: dict[str, Any]) -> str:
    """sha256 of the canonical record without its own event_id."""
    body = {key: value for key, value in record.items() if key != "event_id"}
    return hashlib.sha256(canonical(body)).hexdigest()


# --- field helpers --------------------------------------------------------


def _str(value: Any, min_chars: int = 1) -> bool:
    return isinstance(value, str) and len(value.strip()) >= min_chars


def _str_list(value: Any, min_items: int = 0, min_chars: int = 1) -> bool:
    if not isinstance(value, list):
        return False
    items = cast(list[Any], value)
    return len(items) >= min_items and all(_str(item, min_chars) for item in items)


def _slug(value: Any) -> bool:
    return isinstance(value, str) and _SLUG.match(value) is not None


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.match(value) is not None


def _objects(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    items = cast(list[Any], value)
    if not all(isinstance(item, dict) for item in items):
        return None
    return cast(list[dict[str, Any]], items)


def _ref_errors(value: Any, field: str, *, allow_reference: bool) -> list[str]:
    if not isinstance(value, dict):
        return [f"{field} must be an object"]
    entry = cast(dict[str, Any], value)
    if allow_reference and _str(entry.get("reference")) and entry.get("path") is None:
        return []
    errors: list[str] = []
    if not _str(entry.get("path")):
        errors.append(f"{field}.path is required")
    if not _sha(entry.get("sha256")):
        errors.append(f"{field}.sha256 must be a lowercase hex sha256")
    return errors


def _event_ref_errors(value: Any, field: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{field} must be an object {{system, event_id}}"]
    entry = cast(dict[str, Any], value)
    errors: list[str] = []
    if not _slug(entry.get("system")):
        errors.append(f"{field}.system must be a plugin slug")
    if not _sha(entry.get("event_id")):
        errors.append(f"{field}.event_id must be a sha256")
    return errors


def _envelope_errors(record: dict[str, Any], kind: str) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if record.get("kind") != kind:
        errors.append(f"kind must be {kind!r}")
    if not _slug(record.get("plugin")):
        errors.append("plugin is required")
    if parse_time(record.get("recorded_at")) is None:
        errors.append("recorded_at must be an ISO-8601 timestamp with a timezone")
    sequence = record.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        errors.append("sequence must be a positive integer")
    if not _sha(record.get("prev_event_id")):
        errors.append("prev_event_id must be a sha256 (64 zeros for the first line)")
    writer = record.get("writer")
    if not isinstance(writer, dict):
        errors.append("writer must be an object {plugin, agent?, model?}")
    else:
        entry = cast(dict[str, Any], writer)
        if entry.get("plugin") != record.get("plugin"):
            errors.append("writer.plugin must equal plugin")
        for key in ("agent", "model"):
            if key in entry and not _str(entry[key]):
                errors.append(f"writer.{key} must be a non-empty string when present")
    event = record.get("event_id")
    if not _sha(event):
        errors.append("event_id must be a sha256")
    elif event != compute_event_id(record):
        errors.append("event_id does not match the record content (edited after writing)")
    return errors


# --- impression body ------------------------------------------------------


def _facets_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["facets must be an object {observed, works, concerns, feelings, next_actions}"]
    facets = cast(dict[str, Any], value)
    errors: list[str] = []
    if not _str_list(facets.get("observed"), 2, FACET_MIN_CHARS):
        errors.append(
            f"facets.observed needs at least 2 concrete observations ({FACET_MIN_CHARS}+ chars)"
        )
    if not _str_list(facets.get("works"), 1, FACET_MIN_CHARS):
        errors.append("facets.works needs at least one thing that works")
    concerns = _objects(facets.get("concerns"))
    if concerns is None:
        errors.append("facets.concerns must be a list of objects")
        concerns = []
    if not concerns and not _str(facets.get("no_concerns_reason"), REASON_MIN_CHARS):
        errors.append(
            "facets.concerns is empty: list at least one concern or explain in"
            f" no_concerns_reason ({REASON_MIN_CHARS}+ chars) why nothing worries you"
        )
    ids: list[str] = []
    for index, concern in enumerate(concerns):
        where = f"facets.concerns[{index}]"
        if not _slug(concern.get("id")):
            errors.append(f"{where}.id must be a lowercase slug")
        else:
            ids.append(str(concern["id"]))
        if not _str(concern.get("text"), FACET_MIN_CHARS):
            errors.append(f"{where}.text is required")
        if concern.get("severity") not in SEVERITIES:
            errors.append(f"{where}.severity must be one of {sorted(SEVERITIES)}")
        if not _slug(concern.get("about")):
            errors.append(f"{where}.about must name the plugin that owns it (or 'self')")
        if "anchor" in concern and not _str(concern["anchor"], ANCHOR_MIN_CHARS):
            errors.append(f"{where}.anchor must be a concrete locator when present")
    if len(set(ids)) != len(ids):
        errors.append("facets.concerns ids must be unique")
    feelings = facets.get("feelings")
    if not isinstance(feelings, dict):
        errors.append("facets.feelings must be an object {maker, user}")
    else:
        entry = cast(dict[str, Any], feelings)
        for key in ("maker", "user"):
            if not _str(entry.get(key), FACET_MIN_CHARS):
                errors.append(f"facets.feelings.{key} must say how a {key} would feel")
    if not _str_list(facets.get("next_actions"), 1, FACET_MIN_CHARS):
        errors.append("facets.next_actions needs at least one next step")
    return errors


def _claims_errors(value: Any, min_items: int) -> list[str]:
    claims = _objects(value)
    if claims is None:
        return ["claims must be a list of objects {text, anchor, artifact}"]
    errors: list[str] = []
    if len(claims) < min_items:
        errors.append(
            f"claims needs at least {min_items} grounded claims (a concrete statement plus"
            " the exact token it rests on: part number, net, cavity, dimension, label)"
        )
    for index, claim in enumerate(claims):
        where = f"claims[{index}]"
        if not _str(claim.get("text"), FACET_MIN_CHARS):
            errors.append(f"{where}.text is required")
        if not _str(claim.get("anchor"), ANCHOR_MIN_CHARS):
            errors.append(f"{where}.anchor must be a concrete locator ({ANCHOR_MIN_CHARS}+ chars)")
        if not _str(claim.get("artifact")):
            errors.append(f"{where}.artifact must name the workspace file the claim reads")
    return errors


def _insights_errors(value: Any) -> list[str]:
    insights = _objects(value)
    if insights is None:
        return ["insights must be a list of objects"]
    errors: list[str] = []
    ids: list[str] = []
    for index, insight in enumerate(insights):
        where = f"insights[{index}]"
        if not _slug(insight.get("id")):
            errors.append(f"{where}.id must be a lowercase slug")
        else:
            ids.append(str(insight["id"]))
        if not _str(insight.get("hypothesis"), HYPOTHESIS_MIN_CHARS):
            errors.append(f"{where}.hypothesis must state a testable idea")
        for key in ("proposed_change", "expected_effect", "test"):
            if not _str(insight.get(key), FACET_MIN_CHARS):
                errors.append(f"{where}.{key} is required")
        if not _slug(insight.get("target")):
            errors.append(f"{where}.target must name the plugin that would change")
        revisits = insight.get("revisits")
        if revisits is not None:
            errors.extend(_insight_ref_errors(revisits, f"{where}.revisits"))
            if not _str(insight.get("revisit_reason"), REASON_MIN_CHARS):
                errors.append(f"{where}.revisit_reason must say what new evidence reopens it")
    if len(set(ids)) != len(ids):
        errors.append("insights ids must be unique")
    return errors


def _upstream_errors(value: Any) -> list[str]:
    entries = _objects(value)
    if entries is None:
        return ["upstream must be a list of objects"]
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        where = f"upstream[{index}]"
        errors.extend(_event_ref_errors(entry, where))
        key = (str(entry.get("system")), str(entry.get("event_id")))
        if key in seen:
            errors.append(f"{where} cites the same impression twice")
        seen.add(key)
        if entry.get("disposition") not in DISPOSITIONS:
            errors.append(f"{where}.disposition must be one of {sorted(DISPOSITIONS)}")
        if not _str(entry.get("effect"), EFFECT_MIN_CHARS):
            errors.append(
                f"{where}.effect must say how it changed (or did not change) your design"
                f" ({EFFECT_MIN_CHARS}+ chars)"
            )
        if "concern_ids" in entry and not _str_list(entry["concern_ids"], 1):
            errors.append(f"{where}.concern_ids must list concern ids when present")
    return errors


def impression_body_errors(record: dict[str, Any], *, min_claims: int) -> list[str]:
    errors = impression_errors(record.get("impression"))
    errors.extend(_facets_errors(record.get("facets")))
    errors.extend(_claims_errors(record.get("claims"), min_claims))
    errors.extend(_insights_errors(record.get("insights", [])))
    errors.extend(_upstream_errors(record.get("upstream", [])))
    if record.get("confidence") not in CONFIDENCES:
        errors.append(f"confidence must be one of {sorted(CONFIDENCES)}")
    if not _str_list(record.get("unknowns")):
        errors.append("unknowns must be a list of strings (empty only when truly none)")
    if "delta" in record and not _str(record["delta"], REASON_MIN_CHARS):
        errors.append(f"delta must explain what is new ({REASON_MIN_CHARS}+ chars)")
    claims = _objects(record.get("claims")) or []
    text = str(record.get("impression") or "").casefold()
    anchors = [str(claim.get("anchor") or "").casefold() for claim in claims]
    if claims and not any(anchor and anchor in text for anchor in anchors):
        errors.append(
            "impression must mention at least one claim anchor verbatim, so the prose is"
            " tied to the artifact rather than a template"
        )
    return errors


# --- per-kind validators --------------------------------------------------


def decision_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "decision")
    for key in ("id", "stage"):
        if not _slug(record.get(key)):
            errors.append(f"{key} must be a lowercase slug")
    if not _str(record.get("question"), QUESTION_MIN_CHARS):
        errors.append(f"question must state the decision in at least {QUESTION_MIN_CHARS} chars")
    if not _str_list(record.get("principles"), 1, PRINCIPLE_MIN_CHARS):
        errors.append(
            "principles must cite at least one first principle, law or standard"
            f" (each at least {PRINCIPLE_MIN_CHARS} chars)"
        )
    options = record.get("options")
    names: list[str] = []
    if not isinstance(options, list) or len(cast(list[Any], options)) < 2:
        errors.append("options must list at least two considered alternatives")
    else:
        for index, option in enumerate(cast(list[Any], options)):
            if not isinstance(option, dict):
                errors.append(f"options[{index}] must be an object")
                continue
            entry = cast(dict[str, Any], option)
            if not _str(entry.get("name")):
                errors.append(f"options[{index}].name is required")
            else:
                names.append(str(entry["name"]))
            for side in ("pros", "cons"):
                if not _str_list(entry.get(side), 1):
                    errors.append(f"options[{index}].{side} needs at least one entry")
        if len(set(names)) != len(names):
            errors.append("option names must be unique")
    if record.get("chosen") not in names:
        errors.append("chosen must name one of the options")
    if not _str(record.get("rationale"), RATIONALE_MIN_CHARS):
        errors.append(f"rationale must explain the choice in at least {RATIONALE_MIN_CHARS} chars")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must cite at least one artifact (path+sha256) or reference")
    else:
        for index, entry in enumerate(cast(list[Any], evidence)):
            errors.extend(_ref_errors(entry, f"evidence[{index}]", allow_reference=True))
    for key in ("assumptions", "unknowns"):
        if not _str_list(record.get(key)):
            errors.append(f"{key} must be a list of strings (empty only when truly none)")
    if not _str_list(record.get("risks"), 1):
        errors.append("risks must name at least one residual risk of the chosen option")
    if not _str(record.get("revisit_when")):
        errors.append("revisit_when must name the observation that would reopen the decision")
    if record.get("decided_by") not in DECIDERS:
        errors.append("decided_by must be 'agent' or 'user'")
    refs = record.get("impression_refs", [])
    if not isinstance(refs, list):
        errors.append("impression_refs must be a list")
    else:
        for index, ref in enumerate(cast(list[Any], refs)):
            errors.extend(_event_ref_errors(ref, f"impression_refs[{index}]"))
    insight_refs = record.get("insight_refs", [])
    if not isinstance(insight_refs, list):
        errors.append("insight_refs must be a list")
    else:
        for index, ref in enumerate(cast(list[Any], insight_refs)):
            errors.extend(_insight_ref_errors(ref, f"insight_refs[{index}]"))
    return errors


def stage_impression_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "stage_impression")
    if not _slug(record.get("stage")):
        errors.append("stage must be a lowercase slug")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must list what the stage produced or read (path+sha256)")
    else:
        for index, entry in enumerate(cast(list[Any], artifacts)):
            errors.extend(_ref_errors(entry, f"artifacts[{index}]", allow_reference=False))
    errors.extend(impression_body_errors(record, min_claims=STAGE_CLAIMS_MIN))
    return errors


def vision_review_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "vision_review")
    has_image = _str(record.get("image_path"))
    if has_image and not _sha(record.get("image_sha256")):
        errors.append("image_sha256 must bind the review to the image bytes")
    if not has_image and not _str(record.get("source_event_id")):
        errors.append("a vision review needs image_path or source_event_id")
    if not _str(record.get("model")):
        errors.append("model is required (the vision-capable model that read the image)")
    if not _slug(record.get("checklist")):
        errors.append("checklist must be a lowercase slug")
    if record.get("reviewer") not in REVIEWERS:
        errors.append(f"reviewer must be one of {sorted(REVIEWERS)}")
    findings = _objects(record.get("findings"))
    if findings is None:
        errors.append("findings must be a list of objects")
    else:
        for index, entry in enumerate(findings):
            if not _str(entry.get("category")):
                errors.append(f"findings[{index}].category is required")
            if entry.get("severity") not in SEVERITIES:
                errors.append(f"findings[{index}].severity must be one of {sorted(SEVERITIES)}")
            if not _str(entry.get("note")):
                errors.append(f"findings[{index}].note is required")
    errors.extend(impression_body_errors(record, min_claims=1))
    claims = _objects(record.get("claims")) or []
    lookback = _objects(record.get("lookback"))
    if lookback is None:
        errors.append("lookback must be a list: re-check every claim against the image")
    else:
        covered: set[int] = set()
        for index, entry in enumerate(lookback):
            claim = entry.get("claim")
            if (
                not isinstance(claim, int)
                or isinstance(claim, bool)
                or not 0 <= claim < len(claims)
            ):
                errors.append(f"lookback[{index}].claim must index a claim")
            else:
                covered.add(claim)
            if entry.get("verdict") not in LOOKBACK_VERDICTS:
                errors.append(
                    f"lookback[{index}].verdict must be one of {sorted(LOOKBACK_VERDICTS)}"
                )
            if not _str(entry.get("note"), FACET_MIN_CHARS):
                errors.append(f"lookback[{index}].note must say what you saw on the second look")
        missing = sorted(set(range(len(claims))) - covered)
        if missing:
            errors.append(f"lookback misses claims {missing}; look at the image again for each")
        refuted = [e for e in lookback if e.get("verdict") == "refuted"]
        if refuted and not findings:
            errors.append("a refuted claim must surface as a finding")
    return errors


def upstream_read_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "upstream_read")
    errors.extend(_ref_errors(record.get("artifact"), "artifact", allow_reference=False))
    if not _slug(record.get("producer")):
        errors.append("producer must be a plugin slug")
    refs = record.get("refs")
    if not isinstance(refs, list):
        errors.append("refs must be a list of {system, event_id}")
    else:
        for index, ref in enumerate(cast(list[Any], refs)):
            errors.extend(_event_ref_errors(ref, f"refs[{index}]"))
    if not isinstance(record.get("unresolved"), list):
        errors.append("unresolved must list the problems met while resolving refs")
    return errors


def _insight_ref_errors(value: Any, field: str) -> list[str]:
    errors = _event_ref_errors(value, field)
    if isinstance(value, dict) and not _slug(cast(dict[str, Any], value).get("insight_id")):
        errors.append(f"{field}.insight_id must be a slug")
    return errors


def insight_status_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "insight_status")
    errors.extend(_insight_ref_errors(record.get("insight"), "insight"))
    status = record.get("status")
    if status not in INSIGHT_STATUSES:
        errors.append(f"status must be one of {sorted(INSIGHT_STATUSES)}")
    if not _str(record.get("reason"), REASON_MIN_CHARS):
        errors.append(f"reason must explain the transition ({REASON_MIN_CHARS}+ chars)")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must cite at least one artifact (path+sha256) or reference")
    else:
        for index, entry in enumerate(cast(list[Any], evidence)):
            errors.extend(_ref_errors(entry, f"evidence[{index}]", allow_reference=True))
    gates = _objects(record.get("gate_verdicts", []))
    if gates is None:
        errors.append("gate_verdicts must be a list of {gate, verdict}")
        gates = []
    for index, gate in enumerate(gates):
        if not _str(gate.get("gate")):
            errors.append(f"gate_verdicts[{index}].gate is required")
        if gate.get("verdict") not in GATE_VERDICTS:
            errors.append(f"gate_verdicts[{index}].verdict must be one of {sorted(GATE_VERDICTS)}")
    decisions = record.get("decision_refs", [])
    if not _str_list(decisions):
        errors.append("decision_refs must be a list of decision event ids")
    if status == "adopted":
        if not any(gate.get("verdict") == "pass" for gate in gates):
            errors.append("adopting an insight needs a deterministic gate verdict of pass")
        if not decisions:
            errors.append("adopting an insight needs the decision that adopted it")
    if status == "deferred" and not _str(record.get("revisit_when")):
        errors.append("a deferred insight needs revisit_when")
    return errors


def review_reconcile_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "review_reconcile")
    if not _sha(record.get("image_sha256")):
        errors.append("image_sha256 is required")
    reviews = record.get("reviews")
    if not isinstance(reviews, list) or len(cast(list[Any], reviews)) < 2:
        errors.append("reviews must list at least two vision review event ids")
    elif not all(_sha(item) for item in cast(list[Any], reviews)):
        errors.append("reviews must be event ids")
    if record.get("outcome") not in RECONCILE_OUTCOMES:
        errors.append(f"outcome must be one of {sorted(RECONCILE_OUTCOMES)}")
    round_ = record.get("round")
    if not isinstance(round_, int) or not 1 <= round_ <= MAX_REVIEW_ROUNDS:
        errors.append(f"round must be 1..{MAX_REVIEW_ROUNDS}")
    if not isinstance(record.get("metrics"), dict):
        errors.append("metrics must be an object")
    return errors


def song_receipt_errors(record: dict[str, Any]) -> list[str]:
    errors = _envelope_errors(record, "song_receipt")
    errors.extend(_ref_errors(record.get("delivery"), "delivery", allow_reference=False))
    if not _slug(record.get("song_id")):
        errors.append("song_id must be a slug")
    errors.extend(
        prose_errors(record.get("felt"), "felt", SONG_FEELING_MIN_CHARS, SONG_FEELING_MIN_SENTENCES)
    )
    if not isinstance(record.get("prompted_review"), bool):
        errors.append("prompted_review must be true or false")
    if record.get("prompted_review") is True and not _str(record.get("follow_up"), FACET_MIN_CHARS):
        errors.append("prompted_review needs follow_up: what you will look at again")
    return errors


VALIDATORS = {
    "decision": decision_errors,
    "stage_impression": stage_impression_errors,
    "vision_review": vision_review_errors,
    "upstream_read": upstream_read_errors,
    "insight_status": insight_status_errors,
    "review_reconcile": review_reconcile_errors,
    "song_receipt": song_receipt_errors,
}


def record_errors(kind: str, record: Any) -> list[str]:
    if not isinstance(record, dict):
        return ["record must be a JSON object"]
    return VALIDATORS[kind](cast(dict[str, Any], record))


def song_delivery_errors(value: Any) -> list[str]:
    """Validate a ``liaison/<id>.bard-song.json`` delivery written by bard."""
    if not isinstance(value, dict):
        return ["song delivery must be a JSON object"]
    doc = cast(dict[str, Any], value)
    errors: list[str] = []
    if doc.get("kind") != "bard_song_delivery" or doc.get("schema_version") != 1:
        errors.append("kind must be 'bard_song_delivery' with schema_version 1")
    if not _slug(doc.get("id")):
        errors.append("id must be a slug")
    if doc.get("system") != "bard":
        errors.append("system must be 'bard'")
    if not _slug(doc.get("to")) or doc.get("to") == "bard":
        errors.append("to must name a sister plugin")
    if not _str(doc.get("title")):
        errors.append("title is required")
    if not _str(doc.get("verse"), 40):
        errors.append("verse must carry the song text addressed to the sister")
    errors.extend(_ref_errors(doc.get("song"), "song", allow_reference=False))
    refs = doc.get("impression_refs")
    if not isinstance(refs, list) or not refs:
        errors.append("impression_refs must cite the impressions the song was made from")
    else:
        for index, ref in enumerate(cast(list[Any], refs)):
            errors.extend(_event_ref_errors(ref, f"impression_refs[{index}]"))
    if not _str(doc.get("reason"), FACET_MIN_CHARS):
        errors.append("reason must say why this sister receives the song")
    if parse_time(doc.get("created_at")) is None:
        errors.append("created_at must be an ISO-8601 timestamp with a timezone")
    return errors


# --- logs and chains ------------------------------------------------------


def parse_time(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment.timestamp()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return (object lines, count of malformed lines)."""
    if not path.is_file():
        return [], 0
    records: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value: Any = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            records.append(cast(dict[str, Any], value))
        else:
            malformed += 1
    return records, malformed


def is_legacy(records: list[dict[str, Any]]) -> bool:
    """A log written entirely by VRP v1; the next v2 write archives it."""
    return bool(records) and all(r.get("schema_version") == 1 for r in records)


def chain_errors(records: list[dict[str, Any]], filename: str) -> list[str]:
    """Order, linkage, identity and time checks over one log in file order."""
    errors: list[str] = []
    previous = GENESIS
    last_time: float | None = None
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        where = f"{filename} line {index}"
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{where}: schema_version {record.get('schema_version')!r} in a v2 log")
            break
        if record.get("sequence") != index:
            errors.append(
                f"{where}: sequence {record.get('sequence')!r} (lines missing or reordered)"
            )
        if record.get("prev_event_id") != previous:
            errors.append(f"{where}: prev_event_id does not link to the previous line")
        event = record.get("event_id")
        if not _sha(event) or event != compute_event_id(record):
            errors.append(f"{where}: event_id does not match the content (edited)")
        if isinstance(event, str):
            if event in seen:
                errors.append(f"{where}: duplicate event_id")
            seen.add(event)
            previous = event
        moment = parse_time(record.get("recorded_at"))
        if moment is not None and last_time is not None and moment < last_time:
            errors.append(f"{where}: recorded_at goes back in time")
        if moment is not None:
            last_time = moment
    return errors


def chain_head(records: list[dict[str, Any]]) -> str:
    return str(records[-1].get("event_id")) if records else GENESIS


def verify_log(path: Path, kind: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load a log and return (records, problems); legacy v1 logs have no problems."""
    records, malformed = load_jsonl(path)
    problems: list[str] = []
    if malformed:
        problems.append(f"{path.name}: {malformed} malformed line(s)")
    if is_legacy(records):
        return [], problems
    problems.extend(chain_errors(records, path.name))
    for record in records:
        errors = record_errors(kind, record)
        if errors:
            problems.append(
                f"{path.name} #{record.get('sequence')}: invalid {kind}: " + "; ".join(errors)
            )
    return records, problems


def plugin_logs(
    root: Path, records_dir: Path | None = None
) -> dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]]:
    """Every plugin under ``observations/``: ({kind: records}, integrity problems)."""
    base = records_dir.parent if records_dir is not None else root / OBSERVATIONS_DIR
    result: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]] = {}
    if not base.is_dir():
        return result
    for directory in sorted(p for p in base.iterdir() if p.is_dir() and _slug(p.name)):
        logs: dict[str, list[dict[str, Any]]] = {}
        problems: list[str] = []
        for kind, filename in LOG_FILES.items():
            records, issues = verify_log(directory / filename, kind)
            logs[kind] = records
            problems.extend(issues)
        result[directory.name] = (logs, problems)
    return result


def impression_index(
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for system, (logs, _) in workspace.items():
        for kind in IMPRESSION_KINDS:
            for record in logs.get(kind, []):
                index[(system, str(record.get("event_id")))] = record
    return index


def resolve_ref(
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
    system: str,
    event_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """(record, problem): fail closed on a missing plugin, a broken chain or a missing id."""
    entry = workspace.get(system)
    if entry is None:
        return None, f"no records for {system} in this workspace"
    logs, problems = entry
    if problems:
        return None, f"{system} records fail integrity: {problems[0]}"
    for kind in IMPRESSION_KINDS:
        for record in logs.get(kind, []):
            if record.get("event_id") == event_id:
                return record, None
    return None, f"{system} has no impression {event_id[:12]}"


# --- write-time checks (need the workspace) -------------------------------


def _text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    kept = [p for p in files if not any(part in SKIP_PARTS for part in p.relative_to(path).parts)]
    return kept[:GROUNDING_MAX_FILES]


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > GROUNDING_MAX_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def covers(entry_path: str, relative: str) -> bool:
    entry = entry_path.strip("/").removeprefix("./")
    return relative == entry or relative.startswith(entry + "/") or entry in ("", ".")


def grounding_errors(record: dict[str, Any], root: Path) -> list[str]:
    """Each claim anchor must occur in the text of the artifact it cites.

    Binary artifacts (images, audio) cannot be searched; vision reviews
    ground them through ``lookback``. A stage impression must ground at
    least one claim in text whenever any of its artifacts is text.
    """
    claims = _objects(record.get("claims")) or []
    artifacts = [str(a.get("path")) for a in _objects(record.get("artifacts")) or []]
    if record.get("kind") == "vision_review" and _str(record.get("image_path")):
        artifacts.append(str(record["image_path"]))
    errors: list[str] = []
    grounded = 0
    any_text = False
    for index, claim in enumerate(claims):
        target = str(claim.get("artifact") or "").strip("/").removeprefix("./")
        anchor = str(claim.get("anchor") or "")
        if artifacts and not any(covers(path, target) for path in artifacts):
            errors.append(f"claims[{index}].artifact {target} is not among the bound artifacts")
            continue
        path = (root / target).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            errors.append(f"claims[{index}].artifact {target} is outside the workspace")
            continue
        if not path.exists():
            errors.append(f"claims[{index}].artifact {target} does not exist")
            continue
        texts = [t for t in (_read_text(p) for p in _text_files(path)) if t is not None]
        if not texts:
            continue
        any_text = True
        if any(anchor in text for text in texts):
            grounded += 1
        else:
            errors.append(
                f"claims[{index}].anchor {anchor!r} does not occur in {target}; quote the"
                " exact token you read"
            )
    if record.get("kind") == "stage_impression" and any_text and grounded == 0 and not errors:
        errors.append("no claim is grounded in a text artifact")
    return errors


def novelty_errors(record: dict[str, Any], previous: Iterable[dict[str, Any]]) -> list[str]:
    texts = [str(p.get("impression") or "") for p in previous]
    similarity = max_similarity(str(record.get("impression") or ""), texts)
    if similarity >= NEAR_DUPLICATE_REJECT:
        return [
            f"impression is {similarity:.0%} similar to an earlier one; read the artifact"
            " again and write what is specific to it"
        ]
    if similarity >= NEAR_DUPLICATE_DELTA and not _str(record.get("delta"), REASON_MIN_CHARS):
        return [
            f"impression is {similarity:.0%} similar to an earlier one; add delta"
            f" ({REASON_MIN_CHARS}+ chars) saying what changed since"
        ]
    return []


def upstream_resolution_errors(
    record: dict[str, Any],
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
) -> list[str]:
    errors: list[str] = []
    for index, entry in enumerate(_objects(record.get("upstream", [])) or []):
        target, problem = resolve_ref(
            workspace, str(entry.get("system")), str(entry.get("event_id"))
        )
        if problem:
            errors.append(f"upstream[{index}]: {problem}")
            continue
        assert target is not None
        known = {
            str(c.get("id"))
            for c in _objects(cast(dict[str, Any], target.get("facets") or {}).get("concerns"))
            or []
        }
        unknown = sorted(set(cast(list[str], entry.get("concern_ids") or [])) - known)
        if unknown:
            errors.append(f"upstream[{index}].concern_ids {unknown} are not in that impression")
    return errors


def _artifact_set(record: dict[str, Any]) -> frozenset[tuple[str, str]]:
    pairs = {
        (str(a.get("path")), str(a.get("sha256"))) for a in _objects(record.get("artifacts")) or []
    }
    if _sha(record.get("image_sha256")):
        pairs.add(("image", str(record["image_sha256"])))
    return frozenset(pairs)


def reraise_errors(
    record: dict[str, Any],
    plugin: str,
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
) -> list[str]:
    """Refuse a concern another plugin already disputed while the artifacts are unchanged.

    This is what stops concern ping-pong: a dispute can only be reopened by
    changing the design (new artifact bytes) or by new evidence.
    """
    mine = workspace.get(plugin, ({}, []))[0]
    own = {str(r.get("event_id")): r for k in IMPRESSION_KINDS for r in mine.get(k, [])}
    disputed: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for system, (logs, _) in workspace.items():
        for kind in IMPRESSION_KINDS:
            for other in logs.get(kind, []):
                for entry in _objects(other.get("upstream", [])) or []:
                    if entry.get("system") != plugin or entry.get("disposition") != "disputed":
                        continue
                    earlier = own.get(str(entry.get("event_id")))
                    if earlier is None:
                        continue
                    ids = set(cast(list[str], entry.get("concern_ids") or []))
                    facets = cast(dict[str, Any], earlier.get("facets") or {})
                    for concern in _objects(facets.get("concerns")) or []:
                        if not ids or concern.get("id") in ids:
                            disputed.append((earlier, concern, system))
    errors: list[str] = []
    facets = cast(dict[str, Any], record.get("facets") or {})
    for concern in _objects(facets.get("concerns")) or []:
        text = str(concern.get("text") or "")
        for earlier, old, by in disputed:
            if concern.get("about") != old.get("about"):
                continue
            if _artifact_set(record) != _artifact_set(earlier):
                continue
            if jaccard(shingles(text), shingles(str(old.get("text") or ""))) >= RERAISE_SIMILARITY:
                errors.append(
                    f"concern {concern.get('id')!r} repeats one {by} disputed while the artifacts"
                    " are unchanged; change the design or cite new evidence instead"
                )
                break
    return errors


def insight_history(
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
    ref: dict[str, Any],
) -> list[str]:
    key = (str(ref.get("system")), str(ref.get("event_id")), str(ref.get("insight_id")))
    history = ["proposed"]
    for _, (logs, _) in sorted(workspace.items()):
        for status in logs.get("insight_status", []):
            entry = cast(dict[str, Any], status.get("insight") or {})
            if (
                str(entry.get("system")),
                str(entry.get("event_id")),
                str(entry.get("insight_id")),
            ) == key:
                history.append(str(status.get("status")))
    return history


def insight_write_errors(
    record: dict[str, Any],
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
) -> list[str]:
    ref = cast(dict[str, Any], record.get("insight") or {})
    target, problem = resolve_ref(workspace, str(ref.get("system")), str(ref.get("event_id")))
    if problem:
        return [f"insight: {problem}"]
    assert target is not None
    ids = {str(i.get("id")) for i in _objects(target.get("insights", [])) or []}
    if ref.get("insight_id") not in ids:
        return [f"insight {ref.get('insight_id')!r} is not in that impression"]
    current = insight_history(workspace, ref)[-1]
    status = str(record.get("status"))
    if status not in INSIGHT_TRANSITIONS.get(current, set()):
        return [f"insight cannot move from {current} to {status}"]
    return []


def rejected_insight_errors(
    record: dict[str, Any],
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
) -> list[str]:
    """A rejected hypothesis comes back only through ``revisits`` with a reason."""
    rejected: list[tuple[dict[str, Any], str]] = []
    for system, (logs, _) in workspace.items():
        for kind in IMPRESSION_KINDS:
            for other in logs.get(kind, []):
                for insight in _objects(other.get("insights", [])) or []:
                    ref = {
                        "system": system,
                        "event_id": other.get("event_id"),
                        "insight_id": insight.get("id"),
                    }
                    if insight_history(workspace, ref)[-1] == "rejected":
                        rejected.append((ref, str(insight.get("hypothesis") or "")))
    errors: list[str] = []
    for insight in _objects(record.get("insights", [])) or []:
        if insight.get("revisits") is not None:
            continue
        mine = shingles(str(insight.get("hypothesis") or ""))
        for ref, text in rejected:
            if jaccard(mine, shingles(text)) >= NEAR_DUPLICATE_REJECT:
                errors.append(
                    f"insight {insight.get('id')!r} repeats rejected {ref['system']}:"
                    f"{str(ref['event_id'])[:12]}#{ref['insight_id']};"
                    " use revisits with new evidence"
                )
                break
    return errors


def _error_categories(review: dict[str, Any]) -> set[str]:
    return {
        str(f.get("category"))
        for f in _objects(review.get("findings")) or []
        if f.get("severity") == "error"
    }


def _categories(review: dict[str, Any]) -> set[str]:
    return {str(f.get("category")) for f in _objects(review.get("findings")) or []}


def reviews_agree(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    overlap = jaccard(_categories(left), _categories(right))
    one_sided = sorted(_error_categories(left) ^ _error_categories(right))
    agree = overlap >= RECONCILE_AGREE_JACCARD and not one_sided
    return agree, {"category_jaccard": round(overlap, 4), "one_sided_errors": one_sided}


def reconcile(primary: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic comparison of independent reviews of one image.

    Two reviews agree when their finding categories overlap enough and no
    error-severity category is raised by only one side. A tiebreak review
    (round 2, the last one) settles a split only if it agrees with one of
    the two; error categories then follow the majority. Otherwise the
    outcome is ``unresolved`` and the image counts as unknown, never pass.
    """
    first, second = primary, others[0]
    agree, metrics = reviews_agree(first, second)
    if len(others) == 1:
        return {"round": 1, "outcome": "agree" if agree else "disagree", "metrics": metrics}
    third = others[1]
    sides = [reviews_agree(third, first)[0], reviews_agree(third, second)[0]]
    reviews = [first, second, third]
    union = sorted(set[str]().union(*(_error_categories(r) for r in reviews)))
    majority = [c for c in union if sum(c in _error_categories(r) for r in reviews) >= 2]
    metrics = metrics | {
        "tiebreak_agrees_with": [i for i, s in enumerate(sides) if s],
        "majority_errors": majority,
    }
    return {"round": 2, "outcome": "agree" if any(sides) else "unresolved", "metrics": metrics}


# --- reading across plugins ----------------------------------------------


def concern_status(
    workspace: dict[str, tuple[dict[str, list[dict[str, Any]]], list[str]]],
    system: str,
    record: dict[str, Any],
    concern: dict[str, Any],
) -> str:
    """open, deferred, adopted, disputed or decided for one concern."""
    owner = system if concern.get("about") == "self" else str(concern.get("about"))
    event = str(record.get("event_id"))
    status = "open"
    logs = workspace.get(owner, ({}, []))[0]
    for decision in logs.get("decision", []):
        for ref in _objects(decision.get("impression_refs", [])) or []:
            if ref.get("system") == system and ref.get("event_id") == event:
                return "decided"
    for kind in IMPRESSION_KINDS:
        for later in logs.get(kind, []):
            for entry in _objects(later.get("upstream", [])) or []:
                if entry.get("system") != system or entry.get("event_id") != event:
                    continue
                ids = cast(list[str], entry.get("concern_ids") or [])
                if ids and concern.get("id") not in ids:
                    continue
                disposition = str(entry.get("disposition"))
                if disposition in ("adopted", "disputed"):
                    status = disposition
                elif disposition == "deferred" and status == "open":
                    status = "deferred"
    return status


def acknowledged(
    logs: dict[str, list[dict[str, Any]]], ref: dict[str, Any], after: float | None = None
) -> bool:
    for kind in IMPRESSION_KINDS:
        for record in logs.get(kind, []):
            moment = parse_time(record.get("recorded_at"))
            if after is not None and moment is not None and moment < after:
                continue
            for entry in _objects(record.get("upstream", [])) or []:
                if entry.get("system") == ref.get("system") and entry.get("event_id") == ref.get(
                    "event_id"
                ):
                    return True
    return False


def song_deliveries(root: Path) -> list[tuple[str, dict[str, Any] | None, list[str]]]:
    """(relative path, document or None, problems) for every bard song in liaison/."""
    directory = root / LIAISON_DIR
    found: list[tuple[str, dict[str, Any] | None, list[str]]] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob(f"*{SONG_DELIVERY_SUFFIX}")):
        relative = path.relative_to(root).as_posix()
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            found.append((relative, None, [f"unreadable: {exc}"]))
            continue
        problems = song_delivery_errors(value)
        found.append(
            (relative, cast(dict[str, Any], value) if isinstance(value, dict) else None, problems)
        )
    return found


def received(logs: dict[str, list[dict[str, Any]]], relative: str, digest: str) -> bool:
    return any(
        cast(dict[str, Any], r.get("delivery") or {}).get("path") == relative
        and cast(dict[str, Any], r.get("delivery") or {}).get("sha256") == digest
        for r in logs.get("song_receipt", [])
    )


def digest(root: Path) -> dict[str, Any]:
    """Cross-plugin impression digest: integrity, open concerns, disputes, reads, insights."""
    workspace = plugin_logs(root)
    systems: dict[str, Any] = {}
    concerns: list[dict[str, Any]] = []
    disputes: list[dict[str, Any]] = []
    unread: list[dict[str, Any]] = []
    insights: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for system, (logs, problems) in workspace.items():
        systems[system] = {
            "integrity": "broken" if problems else "ok",
            "problems": problems,
            "counts": {kind: len(items) for kind, items in logs.items()},
            "heads": {kind: chain_head(items) for kind, items in logs.items()},
        }
        if problems:
            continue
        for kind in IMPRESSION_KINDS:
            for record in logs.get(kind, []):
                event = str(record.get("event_id"))
                facets = cast(dict[str, Any], record.get("facets") or {})
                for concern in _objects(facets.get("concerns")) or []:
                    status = concern_status(workspace, system, record, concern)
                    if status in ("open", "deferred", "disputed"):
                        concerns.append(
                            {
                                "ref": f"{system}:{event}#{concern.get('id')}",
                                "system": system,
                                "about": system
                                if concern.get("about") == "self"
                                else concern.get("about"),
                                "severity": concern.get("severity"),
                                "text": concern.get("text"),
                                "anchor": concern.get("anchor"),
                                "status": status,
                                "recorded_at": record.get("recorded_at"),
                            }
                        )
                for entry in _objects(record.get("upstream", [])) or []:
                    if entry.get("disposition") == "disputed":
                        disputes.append(
                            {
                                "by": system,
                                "event_id": event,
                                "target": f"{entry.get('system')}:{entry.get('event_id')}",
                                "effect": entry.get("effect"),
                            }
                        )
                for insight in _objects(record.get("insights", [])) or []:
                    ref = {"system": system, "event_id": event, "insight_id": insight.get("id")}
                    insights.append(
                        {
                            "ref": f"{system}:{event}#{insight.get('id')}",
                            "target": insight.get("target"),
                            "hypothesis": insight.get("hypothesis"),
                            "status": insight_history(workspace, ref)[-1],
                        }
                    )
        for read in logs.get("upstream_read", []):
            after = parse_time(read.get("recorded_at"))
            for ref in _objects(read.get("refs")) or []:
                if not acknowledged(logs, ref, after):
                    unread.append(
                        {"reader": system, "ref": f"{ref.get('system')}:{ref.get('event_id')}"}
                    )
        for item in logs.get("review_reconcile", []):
            if item.get("outcome") != "agree":
                splits.append(
                    {
                        "system": system,
                        "image_sha256": item.get("image_sha256"),
                        "outcome": item.get("outcome"),
                    }
                )
    songs: list[dict[str, Any]] = []
    for relative, doc, problems in song_deliveries(root):
        target = str((doc or {}).get("to") or "")
        logs = workspace.get(target, ({}, []))[0]
        sha = sha256_file(root / relative)
        songs.append(
            {
                "path": relative,
                "to": target,
                "problems": problems,
                "received": received(logs, relative, sha),
            }
        )
    order = {"error": 0, "warning": 1, "info": 2}
    concerns.sort(key=lambda c: (order.get(str(c["severity"]), 3), str(c["recorded_at"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "systems": systems,
        "open_concerns": concerns,
        "disputes": disputes,
        "unacknowledged_reads": unread,
        "insights": insights,
        "review_disagreements": splits,
        "songs": songs,
    }


def search(
    root: Path,
    *,
    query: str = "",
    system: str | None = None,
    kind: str | None = None,
    stage: str | None = None,
    artifact: str | None = None,
    severity: str | None = None,
    open_only: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """Keyword search over every plugin's impressions; broken logs are reported, not mixed in."""
    workspace = plugin_logs(root)
    terms = [t for t in normalized(query).split(" ") if t]
    hits: list[tuple[int, str, dict[str, Any]]] = []
    integrity: dict[str, list[str]] = {}
    for name, (logs, problems) in workspace.items():
        if system and name != system:
            continue
        if problems:
            integrity[name] = problems
            continue
        for record_kind in IMPRESSION_KINDS:
            if kind and record_kind != kind:
                continue
            for record in logs.get(record_kind, []):
                if stage and record.get("stage") != stage:
                    continue
                paths = [str(a.get("path")) for a in _objects(record.get("artifacts")) or []]
                shas = [str(a.get("sha256")) for a in _objects(record.get("artifacts")) or []]
                if record.get("image_path"):
                    paths.append(str(record["image_path"]))
                    shas.append(str(record.get("image_sha256")))
                if artifact and not any(artifact in p for p in paths) and artifact not in shas:
                    continue
                facets = cast(dict[str, Any], record.get("facets") or {})
                concerns = _objects(facets.get("concerns")) or []
                if severity and not any(c.get("severity") == severity for c in concerns):
                    continue
                statuses = [concern_status(workspace, name, record, c) for c in concerns]
                if open_only and not any(s in ("open", "deferred", "disputed") for s in statuses):
                    continue
                blob = normalized(json.dumps(record, ensure_ascii=False))
                score = sum(blob.count(term) for term in terms)
                if terms and score == 0:
                    continue
                summary = {
                    "system": name,
                    "kind": record_kind,
                    "event_id": record.get("event_id"),
                    "recorded_at": record.get("recorded_at"),
                    "stage": record.get("stage"),
                    "artifacts": paths,
                    "impression": record.get("impression"),
                    "concerns": [
                        {**c, "status": s} for c, s in zip(concerns, statuses, strict=True)
                    ],
                    "insights": record.get("insights", []),
                }
                hits.append((score, str(record.get("recorded_at")), summary))
    hits.sort(key=lambda item: (-item[0], _neg_time(item[1])))
    return {
        "verdict": "unknown" if integrity else "pass",
        "integrity_problems": integrity,
        "results": [item[2] for item in hits[: max(limit, 0)]],
    }


def _neg_time(value: str) -> float:
    moment = parse_time(value)
    return -(moment or 0.0)


def recall(root: Path, plugin: str, limit: int = 8) -> str:
    """Short SessionStart briefing: what sisters worry about for this plugin, songs waiting."""
    summary = digest(root)
    lines: list[str] = []
    broken = [name for name, info in summary["systems"].items() if info["integrity"] != "ok"]
    if broken:
        lines.append(f"- records with broken integrity (treat as unknown): {', '.join(broken)}")
    for concern in summary["open_concerns"]:
        if concern["about"] != plugin or len(lines) >= limit:
            continue
        lines.append(
            f"- {concern['severity']} from {concern['system']} ({concern['status']}):"
            f" {concern['text']} [{concern['ref']}]"
        )
    for read in summary["unacknowledged_reads"]:
        if read["reader"] == plugin and len(lines) < limit:
            lines.append(f"- impression {read['ref']} came with an import and is still unanswered")
    for song in summary["songs"]:
        if song["to"] == plugin and not song["received"] and len(lines) < limit:
            lines.append(f"- bard sent you a song: {song['path']} (record a song receipt)")
    if not lines:
        return ""
    return (
        f"[{plugin}] Sister impressions waiting for you (VRP v2). Cite each one you act on in"
        " the upstream field of your next impression with adopted/deferred/disputed/noted:\n"
        + "\n".join(lines)
    )


# --- policy and changed artifacts ----------------------------------------


def load_policy(plugin_root: Path) -> dict[str, Any]:
    path = plugin_root / "hooks" / "records-policy.json"
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object")
    policy = cast(dict[str, Any], value)
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    if not _str(policy.get("plugin")) or not _str(policy.get("records_dir")):
        raise ValueError(f"{path}: plugin and records_dir are required")
    if not _str_list(policy.get("artifact_globs"), 1):
        raise ValueError(f"{path}: artifact_globs must list at least one glob")
    if not _str_list(policy.get("dual_review_globs", [])):
        raise ValueError(f"{path}: dual_review_globs must be a list of globs")
    return policy


def matches(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def changed_artifacts(root: Path, policy: dict[str, Any], since: float) -> list[str]:
    """Workspace-relative files matching artifact_globs modified at or after `since`."""
    records_dir = str(policy["records_dir"]).strip("/")
    ignore = [*cast(list[str], policy.get("ignore_globs") or []), f"{OBSERVATIONS_DIR}/**"]
    found: set[str] = set()
    for pattern in cast(list[str], policy["artifact_globs"]):
        for path in root.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            parts = path.relative_to(root).parts
            if any(part in SKIP_PARTS for part in parts):
                continue
            if relative.startswith(records_dir + "/") or matches(relative, ignore):
                continue
            if relative.endswith(SONG_DELIVERY_SUFFIX):
                continue
            try:
                if path.stat().st_mtime >= since:
                    found.add(relative)
            except OSError:
                continue
    return sorted(found)
