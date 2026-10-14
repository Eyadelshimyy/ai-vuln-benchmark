"""Interchange envelope v3: a provenance sidecar beside each sister artifact.

The producer writes ``<artifact>.provenance.json`` next to the artifact it
hands to a sister: the artifact's sha256, the sources it was projected from
(workspace-relative path + sha256) and the producer's VRP decisions whose
evidence cites those sources at their current bytes. The consumer checks the
sidecar against the bytes it pinned at import, the sources on disk and the
producer's intact decision log. Shared verbatim across VibeBB agents.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from . import _vrp

ENVELOPE_VERSION = 3
SIDECAR_SUFFIX = ".provenance.json"
Status = Literal["pass", "fail", "unknown"]


def sidecar_path(artifact: Path) -> Path:
    """``x.fpga-regmap.json`` -> ``x.fpga-regmap.provenance.json``."""
    return artifact.with_name(artifact.name.removesuffix(".json") + SIDECAR_SUFFIX)


def _source_key(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _decision_log(producer: str, root: Path) -> Path:
    return root / _vrp.OBSERVATIONS_DIR / producer / "decisions.jsonl"


def _cites(record: dict[str, Any], cited: set[tuple[str, str]]) -> bool:
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return False
    for entry in cast(list[object], evidence):
        if isinstance(entry, dict):
            item = cast(dict[str, object], entry)
            if (str(item.get("path")), str(item.get("sha256"))) in cited:
                return True
    return False


def decision_refs(producer: str, sources: Iterable[tuple[str, str]], root: Path) -> list[str]:
    """Decision event ids whose evidence cites a source at its current sha256."""
    log = _decision_log(producer, root)
    if not log.is_file():
        return []
    records, problems = _vrp.verify_log(log, "decision")
    if problems:
        raise ValueError(f"{producer} decisions fail integrity: {problems[0]}")
    cited = set(sources)
    return sorted({str(r.get("event_id")) for r in records if _cites(r, cited)})


def write_envelope(
    artifact: Path, producer: str, sources: Iterable[Path], root: Path
) -> dict[str, Any]:
    """Write the v3 sidecar for ``artifact``; a broken decision log refuses it."""
    pins = sorted({(_source_key(p, root), _vrp.sha256_file(p)) for p in sources})
    body: dict[str, Any] = {
        "envelope_version": ENVELOPE_VERSION,
        "producer": producer,
        "artifact": {"name": artifact.name, "sha256": _vrp.sha256_file(artifact)},
        "sources": [{"path": path, "sha256": digest} for path, digest in pins],
        "decision_refs": decision_refs(producer, pins, root),
    }
    sidecar_path(artifact).write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return body


def _read(side: Path) -> dict[str, Any]:
    document: object = json.loads(side.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("not a JSON object")
    body = cast(dict[str, Any], document)
    artifact = body.get("artifact")
    sources = body.get("sources")
    refs = body.get("decision_refs")
    if (
        not isinstance(artifact, dict)
        or not isinstance(sources, list)
        or not isinstance(refs, list)
        or not all(isinstance(ref, str) for ref in cast(list[object], refs))
        or not all(
            isinstance(s, dict)
            and isinstance(cast(dict[str, object], s).get("path"), str)
            and isinstance(cast(dict[str, object], s).get("sha256"), str)
            for s in cast(list[object], sources)
        )
    ):
        raise ValueError("needs artifact{name,sha256}, sources[{path,sha256}], decision_refs[]")
    return body


def check_envelope(
    artifact: Path, producer: str, root: Path, pinned_sha256: str | None = None
) -> tuple[Status, str]:
    """(status, detail) for the sidecar beside an imported sister artifact."""
    side = sidecar_path(artifact)
    try:
        body = _read(side)
    except FileNotFoundError:
        return "unknown", f"no envelope v3 provenance ({side.name})"
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return "fail", f"malformed provenance {side.name}: {exc}"
    failed: list[str] = []
    unknown: list[str] = []
    if body.get("envelope_version") != ENVELOPE_VERSION:
        failed.append(f"envelope_version is not {ENVELOPE_VERSION}")
    if body.get("producer") != producer:
        failed.append(f"producer is {body.get('producer')!r}, expected {producer!r}")
    expected = pinned_sha256
    if expected is None:
        try:
            expected = _vrp.sha256_file(artifact)
        except OSError as exc:
            return "unknown", f"artifact unreadable: {exc}"
    pinned = cast(dict[str, object], body["artifact"])
    if pinned.get("name") != artifact.name or pinned.get("sha256") != expected:
        failed.append("provenance describes other artifact bytes")
    for source in cast(list[dict[str, str]], body["sources"]):
        path = Path(source["path"])
        path = path if path.is_absolute() else root / path
        try:
            current = _vrp.sha256_file(path)
        except OSError:
            unknown.append(f"source unreadable: {source['path']}")
            continue
        if current != source["sha256"]:
            failed.append(f"source changed since export: {source['path']}")
    refs = cast(list[str], body["decision_refs"])
    if refs:
        log = _decision_log(producer, root)
        if not log.is_file():
            unknown.append(f"no {producer} decision log for {len(refs)} decision_refs")
        else:
            records, problems = _vrp.verify_log(log, "decision")
            if problems:
                failed.append(f"{producer} decisions fail integrity: {problems[0]}")
            else:
                known = {str(r.get("event_id")) for r in records}
                missing = sorted(ref for ref in refs if ref not in known)
                if missing:
                    failed.append(f"decision_refs not in {producer} log: {', '.join(missing)}")
    if failed:
        return "fail", "; ".join(failed)
    if unknown:
        return "unknown", "; ".join(unknown)
    return "pass", f"{len(body['sources'])} sources, {len(refs)} decisions"
