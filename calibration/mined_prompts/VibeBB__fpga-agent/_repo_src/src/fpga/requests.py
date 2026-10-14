"""Change requests from the FPGA agent to a sibling (``*.fpga-request.json``).

The FPGA agent never edits a sibling's inputs. When the pin map needs a
circuit change (a different ball, a pull resistor, a level shifter, a
clock source) or a firmware-side change, it writes a request the owning
agent triages. Schema v2 binds the request to hashed inputs and to the
decision records that justify it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .interchange import sha256_file
from .records import decision_event_ids
from .workspace import workspace_path, workspace_root

Target = Literal[
    "bard",
    "circuit",
    "dashboard",
    "doc",
    "firmware",
    "mech",
    "prodeng",
    "sim",
    "wire",
    "ux-creator",
]

_SHA256: str = r"^[0-9a-f]{64}$"


class RequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, description="Workspace-relative path")
    sha256: str = Field(pattern=_SHA256)


class FpgaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    system: Literal["fpga"] = "fpga"
    artifact_kind: Literal["fpga_request"] = "fpga_request"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    design: str
    target: Target
    risk: Literal["low", "high"]
    change: str = Field(min_length=8)
    rationale: str = Field(min_length=8)
    nets: list[str] = Field(default_factory=list[str])
    failing_checks: list[str] = Field(default_factory=list[str])
    contract_sha256: str
    inputs: list[RequestInput] = Field(min_length=1)
    decision_refs: list[str] = Field(min_length=1)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "request"


def write_request(
    contract_path: Path,
    design: str,
    out_dir: Path,
    *,
    target: str,
    risk: str,
    change: str,
    rationale: str,
    nets: list[str],
    failing_checks: list[str],
    extra_inputs: list[str] | None = None,
    decision_refs: list[str] | None = None,
    root: Path | None = None,
) -> tuple[FpgaRequest, Path]:
    base = (root or workspace_root()).resolve()
    decision_refs = decision_refs or []
    known = decision_event_ids(base)
    for ref in decision_refs:
        if ref not in known:
            raise ValueError(f"decision_ref {ref} is not a decision event_id")
    if not decision_refs:
        raise ValueError("a request needs at least one decision_ref")
    contract_in = workspace_path(contract_path, base)
    inputs = [
        RequestInput(
            path=contract_in.relative_to(base).as_posix(),
            sha256=sha256_file(contract_in),
        )
    ]
    for value in extra_inputs or []:
        path = workspace_path(value, base)
        if not path.is_file():
            raise ValueError(f"input does not exist: {value}")
        inputs.append(
            RequestInput(path=path.relative_to(base).as_posix(), sha256=sha256_file(path))
        )
    request = FpgaRequest.model_validate(
        {
            "id": f"{target}-{slug(change)}",
            "design": design,
            "target": target,
            "risk": risk,
            "change": change,
            "rationale": rationale,
            "nets": nets,
            "failing_checks": failing_checks,
            "contract_sha256": sha256_file(contract_path),
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "decision_refs": decision_refs,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{design}.{request.id}.fpga-request.json"
    path.write_text(
        json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return request, path
