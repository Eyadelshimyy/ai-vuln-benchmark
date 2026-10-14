"""SLP v2 liaison: the FPGA side of UX-creator's request protocol.

UX-creator drops ``liaison/<id>.ux-request.json`` files in the workspace;
the FPGA agent answers with ``liaison/<id>.ux-response.json``. These are
strict local mirrors of the UX producer's schema — the UX producer
validates job-id citations against its own contract, this mirror cannot
(documented limitation).

Refusals are fail-closed: a malformed request, a missing artifact, a
stale input, or a ``done`` answer without evidence raises ``ValueError``
(the MCP tool reports ``isError`` and the CLI exits 1).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from . import _vrp
from .records import (
    decision_event_ids,
    impression_event_ids,
    sha256_file,
    tree_sha256,
)
from .records import records_dir as _heads_records_dir
from .workspace import workspace_path, workspace_root

SCHEMA_VERSION = 2
REQUEST_SUFFIX = ".ux-request.json"
RESPONSE_SUFFIX = ".ux-response.json"
_ID = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_SHA256 = r"^[0-9a-f]{64}$"

Target = Literal[
    "bard",
    "circuit",
    "dashboard",
    "doc",
    "firmware",
    "fpga",
    "mech",
    "prodeng",
    "sim",
    "wire",
]
Stage = Literal[
    "requirements", "design", "manufacturing_handoff", "build", "evaluation", "revision"
]
Risk = Literal["low", "high"]
Status = Literal["accepted", "in_progress", "done", "rejected", "deferred", "needs_info"]
Verdict = Literal["pass", "fail", "unknown"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputRef(_Strict):
    path: str = Field(min_length=1, description="Workspace-relative path")
    sha256: str = Field(pattern=_SHA256)


class GateVerdict(_Strict):
    gate: str = Field(min_length=1)
    verdict: Verdict


class UxRequest(_Strict):
    """A job UX-creator hands to a sibling plugin."""

    schema_version: Literal[2] = SCHEMA_VERSION
    system: Literal["ux-creator"] = "ux-creator"
    id: str = Field(pattern=_ID)
    target_agent: Target
    stage: Stage
    risk: Risk
    purpose: str = Field(min_length=20)
    rationale: str = Field(min_length=1)
    requested_changes: list[str] = Field(min_length=1)
    inputs: list[InputRef] = Field(default_factory=list[InputRef])
    expected_deliverables: list[str] = Field(min_length=1)
    acceptance: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list[str])
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _lists_have_items(self) -> UxRequest:
        for name in ("requested_changes", "expected_deliverables", "acceptance"):
            if any(not item.strip() for item in getattr(self, name)):
                raise ValueError(f"{name} items must be non-empty")
        if self.risk == "high" and len(self.rationale.strip()) < 20:
            raise ValueError("high-risk requests need a rationale of at least 20 characters")
        return self


class UxResponse(_Strict):
    """The sibling's answer; written as ``liaison/<id>.ux-response.json``."""

    schema_version: Literal[2] = SCHEMA_VERSION
    system: Literal["ux-creator"] = "ux-creator"
    request: str = Field(pattern=_ID)
    responder: Target
    status: Status
    reason: str = ""
    input_hashes: dict[str, str] = Field(default_factory=dict[str, str])
    artifacts: list[InputRef] = Field(default_factory=list[InputRef])
    gate_verdicts: list[GateVerdict] = Field(default_factory=list[GateVerdict])
    decision_refs: list[str] = Field(default_factory=list[str])
    impression_refs: list[str] = Field(default_factory=list[str])
    questions_for_user: list[str] = Field(default_factory=list[str])
    records_heads: dict[str, str] = Field(default_factory=dict[str, str])
    responded_at: AwareDatetime

    @model_validator(mode="after")
    def _consistent(self) -> UxResponse:
        if self.status not in ("accepted", "in_progress") and len(self.reason.strip()) < 20:
            raise ValueError(f"status {self.status} needs a reason of at least 20 characters")
        if self.status == "done" and any(v.verdict != "pass" for v in self.gate_verdicts):
            raise ValueError("done cannot carry a fail or unknown gate verdict")
        if self.status == "needs_info" and not self.questions_for_user:
            raise ValueError("needs_info needs at least one question for the user")
        return self


def liaison_dir(root: Path) -> Path:
    return root / "liaison"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return cast(dict[str, Any], data)


def _request_file_id(path: Path) -> str:
    return path.name.removesuffix(REQUEST_SUFFIX)


def _read_request(path: Path) -> UxRequest:
    request = UxRequest.model_validate(_load_json(path))
    if request.id != _request_file_id(path):
        raise ValueError(f"request id {request.id} does not match file stem {path.name}")
    return request


def _valid_responses(directory: Path) -> tuple[dict[str, UxResponse], list[dict[str, str]]]:
    """request id -> response, plus every malformed response file."""
    found: dict[str, UxResponse] = {}
    malformed: list[dict[str, str]] = []
    if not directory.is_dir():
        return found, malformed
    for path in sorted(directory.glob(f"*{RESPONSE_SUFFIX}")):
        try:
            response = UxResponse.model_validate(_load_json(path))
        except Exception as exc:
            malformed.append({"path": str(path), "error": str(exc)})
            continue
        if response.request != path.name.removesuffix(RESPONSE_SUFFIX):
            malformed.append(
                {
                    "path": str(path),
                    "error": f"response request {response.request} does not match file stem",
                }
            )
            continue
        found[response.request] = response
    return found, malformed


def _input_stale(request: UxRequest, root: Path, response: UxResponse | None) -> list[str]:
    stale: list[str] = []
    for item in request.inputs:
        try:
            path = workspace_path(item.path, root)
        except ValueError:
            stale.append(item.path)
            continue
        current = sha256_file(path) if path.is_file() else None
        if current != item.sha256 or (
            response is not None and response.input_hashes.get(item.path, current) != current
        ):
            stale.append(item.path)
    return stale


def inbox(workspace: Path | str | None = None) -> dict[str, Any]:
    """Triage liaison requests for the FPGA agent."""
    root = workspace_path(str(workspace)) if workspace is not None else workspace_root()
    directory = liaison_dir(root)
    responses, malformed = _valid_responses(directory)
    requests: list[dict[str, Any]] = []
    counts = {"new": 0, "answered": 0, "stale": 0, "blocked": 0}
    if directory.is_dir():
        for path in sorted(directory.glob(f"*{REQUEST_SUFFIX}")):
            try:
                raw = _load_json(path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                malformed.append({"path": str(path), "error": str(exc)})
                continue
            target = raw.get("target_agent")
            try:
                request = _read_request(path)
            except Exception as exc:
                if target in (None, "fpga"):
                    malformed.append({"path": str(path), "error": str(exc)})
                continue
            if request.target_agent != "fpga":
                continue
            response = responses.get(request.id)
            stale_inputs = _input_stale(request, root, response)
            changed: list[str] = []
            if response is not None:
                for artifact in response.artifacts:
                    try:
                        produced = workspace_path(artifact.path, root)
                    except ValueError:
                        changed.append(artifact.path)
                        continue
                    if not produced.exists() or tree_sha256(produced) != artifact.sha256:
                        changed.append(artifact.path)
            problems = [f"input changed or missing: {p}" for p in stale_inputs]
            problems += [f"artifact changed or missing: {p}" for p in changed]
            if stale_inputs or changed:
                state = "stale"
            elif response is not None and response.responder == "fpga":
                state = "answered"
            elif any(dep not in responses for dep in request.depends_on):
                state = "blocked"
                problems += [
                    f"depends_on {dep} has no valid response"
                    for dep in request.depends_on
                    if dep not in responses
                ]
            else:
                state = "new"
            counts[state] += 1
            requests.append(
                {
                    "id": request.id,
                    "path": str(path),
                    "stage": request.stage,
                    "risk": request.risk,
                    "state": state,
                    "response_status": response.status if response else None,
                    "depends_on": request.depends_on,
                    "stale_inputs": stale_inputs,
                    "changed_artifacts": changed,
                    "problems": problems,
                }
            )
    return {"verdict": "pass", "requests": requests, "malformed": malformed, "counts": counts}


def _records_heads(root: Path) -> tuple[dict[str, str], list[str]]:
    """Chain head per intact VRP log, plus the kinds whose chain is broken."""
    heads: dict[str, str] = {}
    broken: list[str] = []
    for kind, filename in _vrp.LOG_FILES.items():
        records, problems = _vrp.verify_log(_heads_records_dir(root) / filename, kind)
        if problems:
            broken.append(kind)
        else:
            heads[kind] = _vrp.chain_head(records)
    return heads, broken


def respond(
    workspace: Path | str | None,
    request_id: str,
    status: str,
    *,
    reason: str = "",
    artifacts: list[str] | None = None,
    gate_verdicts: list[dict[str, str]] | None = None,
    gate_report: str | Path | None = None,
    decision_refs: list[str] | None = None,
    impression_refs: list[str] | None = None,
    questions_for_user: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``liaison/<id>.ux-response.json`` after validating the refusal rules."""
    root = workspace_path(str(workspace)) if workspace is not None else workspace_root()
    directory = liaison_dir(root)
    request_path = directory / f"{request_id}{REQUEST_SUFFIX}"
    if not request_path.is_file():
        raise ValueError(f"no liaison request {request_id}")
    try:
        request = _read_request(request_path)
    except Exception as exc:
        raise ValueError(f"request {request_id} is malformed: {exc}") from exc
    if request.target_agent != "fpga":
        raise ValueError(f"request {request_id} targets {request.target_agent}, not fpga")

    warnings: list[str] = []
    decision_refs = decision_refs or []
    impression_refs = impression_refs or []
    decisions = decision_event_ids(root)
    impressions = impression_event_ids(root)
    for ref in decision_refs:
        if ref not in decisions:
            raise ValueError(f"decision_ref {ref} is not a decision event_id")
    for ref in impression_refs:
        if ref not in impressions:
            raise ValueError(f"impression_ref {ref} is not an impression or vision-review event_id")

    stale = _input_stale(request, root, None)
    if status in ("done", "accepted", "in_progress") and stale:
        raise ValueError(
            f"request inputs changed or are missing ({', '.join(stale)}); "
            "ask UX to reissue the request, or answer needs_info"
        )

    artifact_refs: list[InputRef] = []
    for value in artifacts or []:
        path = workspace_path(value, root)
        if not path.exists():
            raise ValueError(f"artifact does not exist: {value}")
        artifact_refs.append(
            InputRef(path=path.relative_to(root).as_posix(), sha256=tree_sha256(path))
        )

    verdicts: list[GateVerdict] = []
    if gate_report is not None:
        report_path = workspace_path(str(gate_report), root)
        try:
            report = _load_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"gate report unreadable: {exc}") from exc
        for check in cast(list[Any], report.get("checks", [])):
            if not isinstance(check, dict):
                continue
            item = cast(dict[str, Any], check)
            if item.get("status") == "not_applicable":
                continue
            verdict: Verdict = (
                "pass"
                if item.get("status") == "pass"
                else ("fail" if item.get("status") == "fail" else "unknown")
            )
            verdicts.append(GateVerdict(gate=str(item.get("id", "?")), verdict=verdict))
        if report_path.is_file():
            artifact_refs.append(
                InputRef(
                    path=report_path.relative_to(root).as_posix(),
                    sha256=sha256_file(report_path),
                )
            )
    for item in gate_verdicts or []:
        verdicts.append(GateVerdict.model_validate(item))

    if status == "done":
        if any(v.verdict != "pass" for v in verdicts):
            raise ValueError(
                "done cannot carry a fail or unknown verdict; "
                "answer needs_info or rejected with a reason"
            )
        if not artifact_refs or not verdicts or not decision_refs or not impression_refs:
            raise ValueError(
                "done needs at least one artifact, one gate verdict, one decision_ref "
                "and one impression_ref"
            )

    input_hashes: dict[str, str] = {}
    for item in request.inputs:
        try:
            path = workspace_path(item.path, root)
        except ValueError:
            warnings.append(f"input {item.path} is outside the workspace")
            continue
        if path.is_file():
            input_hashes[item.path] = sha256_file(path)
        else:
            warnings.append(f"input {item.path} is missing")

    heads, broken_logs = _records_heads(root)
    if status == "done" and broken_logs:
        raise ValueError(f"refusing 'done': VRP logs fail integrity ({', '.join(broken_logs)})")

    response = UxResponse(
        request=request.id,
        responder="fpga",
        status=cast(Status, status),
        reason=reason,
        input_hashes=input_hashes,
        artifacts=artifact_refs,
        gate_verdicts=verdicts,
        decision_refs=decision_refs,
        impression_refs=impression_refs,
        questions_for_user=questions_for_user or [],
        records_heads=heads,
        responded_at=datetime.now(UTC),
    )
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{request.id}{RESPONSE_SUFFIX}"
    out_path.write_text(
        json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "verdict": "pass",
        "written": str(out_path),
        "sha256": sha256_file(out_path),
        "response": response.model_dump(mode="json"),
        "warnings": warnings,
    }
