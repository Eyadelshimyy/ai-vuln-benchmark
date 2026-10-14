"""Expose the fpga entry points over a stdio MCP transport.

Every tool returns the same JSON payload as the CLI; the transport never
judges the design itself. Board programming is deliberately not exposed.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from mcp import types
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from . import __version__, service
from .records import (
    DecisionInput,
    InsightStatusInput,
    ReadInput,
    ReconcileInput,
    SongReceiptInput,
    StageImpressionInput,
    VisionReviewInput,
    record_decision,
    record_impression,
    record_insight,
    record_read,
    record_reconcile,
    record_song_receipt,
    record_vision_review,
    records_digest,
    records_search,
    records_summary,
)
from .workspace import workspace_path

server: Server = Server(f"fpga-mcp/{__version__}")

_CONTRACT = {"contract_path": {"type": "string"}}
_OUT = {"out_dir": {"type": "string"}}
_STRINGS = {"type": "array", "items": {"type": "string"}}


def _schema(properties: Mapping[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


TOOLS: dict[str, tuple[str, dict[str, object], bool]] = {
    "fpga_doctor": ("Probe the open-source FPGA toolchain", _schema({}, []), True),
    "fpga_validate": (
        "Validate a <name>.fpga.json contract and resolve its device profile",
        _schema(_CONTRACT, ["contract_path"]),
        True,
    ),
    "fpga_check": (
        "Static gates: contract, library provenance, pins, constraints, netlist match",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_gates": (
        "All gates: static + lint, simulation, formal, synthesis, P&R, timing, "
        "utilization, bitstream",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_constraints": (
        "Regenerate the contract's PCF/LPF/CST constraint file",
        _schema(_CONTRACT, ["contract_path"]),
        False,
    ),
    "fpga_pinmap_export": (
        "Export <name>.fpga-pinmap.json for electrical-circuit-agent",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_regmap_export": (
        "Write the HDL register constants and <name>.fpga-regmap.json for firmware-agent",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_lint": (
        "Analyse and elaborate the design top (NVC for VHDL, Verilator for Verilog)",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_sim": (
        "Run one declared simulation (NVC or Icarus Verilog)",
        _schema(
            {**_CONTRACT, **_OUT, "simulation": {"type": "string"}}, ["contract_path", "simulation"]
        ),
        False,
    ),
    "fpga_formal": (
        "Run one declared SymbiYosys proof, BMC or cover",
        _schema({**_CONTRACT, **_OUT, "formal": {"type": "string"}}, ["contract_path", "formal"]),
        False,
    ),
    "fpga_build": (
        "Synthesis, place and route and bitstream packing (advisory; gates are authoritative)",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_production_export": (
        "Export <name>.fpga-production.json (the gated bitstream and loader options) "
        "for production-engineering-agent; never touches hardware",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_request": (
        "Write a change request (*.fpga-request.json) to a sibling agent",
        _schema(
            {
                **_CONTRACT,
                **_OUT,
                "target": {"type": "string"},
                "risk": {"type": "string", "enum": ["low", "high"]},
                "change": {"type": "string"},
                "rationale": {"type": "string"},
                "nets": _STRINGS,
                "failing_checks": _STRINGS,
                "inputs": _STRINGS,
                "decision_refs": _STRINGS,
            },
            ["contract_path", "target", "risk", "change", "rationale", "decision_refs"],
        ),
        False,
    ),
    "fpga_sim_thermal_request": (
        "Write <name>.thermal.sim.json and its *.sim-request.json for simulation-agent "
        "(junction temperature of the FPGA package from authored thermal facts)",
        _schema({**_CONTRACT, **_OUT}, ["contract_path"]),
        False,
    ),
    "fpga_sim_thermal_check": (
        "Read simulation-agent's hash-bound thermal answer; stale or tampered answers fail, "
        "missing or deferred ones are unknown",
        _schema(_CONTRACT, ["contract_path"]),
        True,
    ),
    "fpga_profile": (
        "List bundled device profiles, or show one (package pins, resources)",
        _schema({"profile": {"type": "string"}}, []),
        True,
    ),
    "fpga_record_decision": (
        "Record a non-trivial design decision (question, first principles, "
        "options, chosen option, 200+ char rationale, evidence, risks)",
        DecisionInput.model_json_schema(),
        False,
    ),
    "fpga_record_impression": (
        "Record a long-form stage impression (400+ chars, 3+ sentences) bound "
        "to the stage artifacts by sha256",
        StageImpressionInput.model_json_schema(),
        False,
    ),
    "fpga_record_vision_review": (
        "Record findings plus a long-form impression after looking at an image; "
        "bind to image_path or a vision source_event_id",
        VisionReviewInput.model_json_schema(),
        False,
    ),
    "fpga_records_status": (
        "Counts of decision / impression / vision-review records and the last "
        "Stop-hook verdict listing records this session still owes",
        _schema({}, []),
        True,
    ),
    "fpga_record_reconcile": (
        "Compare independent reviews of one image: [primary, blind] (round 1) or "
        "[primary, blind, tiebreak] (round 2, the last); a round-2 split stays unknown",
        ReconcileInput.model_json_schema(),
        False,
    ),
    "fpga_record_insight": (
        "Move an impression insight through tried/adopted/rejected/deferred/superseded; "
        "a rejected insight returns only with revisits and new evidence",
        InsightStatusInput.model_json_schema(),
        False,
    ),
    "fpga_record_song_receipt": (
        "Receive a bard song addressed to fpga; a receipt never feeds a gate and "
        "never asks bard for another song",
        SongReceiptInput.model_json_schema(),
        False,
    ),
    "fpga_record_read": (
        "Record that a sister artifact was taken in; resolves the impressions bound to it",
        ReadInput.model_json_schema(),
        False,
    ),
    "fpga_records_digest": (
        "Family-wide impression digest: integrity per plugin, open concerns, disputes, "
        "unanswered reads, insights, split reviews and undelivered songs",
        _schema({}, []),
        True,
    ),
    "fpga_records_search": (
        "Search every plugin's impressions by text, plugin, stage, artifact, severity or "
        "open concerns; logs with broken integrity are listed separately",
        _schema(
            {
                "query": {"type": "string"},
                "system": {"type": "string"},
                "kind": {"type": "string", "enum": ["stage_impression", "vision_review"]},
                "stage": {"type": "string"},
                "artifact": {"type": "string"},
                "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                "open_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            [],
        ),
        True,
    ),
    "fpga_ux_inbox": (
        "Triage liaison/*.ux-request.json files addressed to the FPGA agent "
        "(new / answered / stale / blocked, malformed files)",
        _schema({"workspace": {"type": "string"}}, []),
        True,
    ),
    "fpga_ux_respond": (
        "Answer a UX-creator request with liaison/<id>.ux-response.json "
        "(refuses stale inputs, unverifiable refs, or an unproven done)",
        _schema(
            {
                "workspace": {"type": "string"},
                "request": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        "accepted",
                        "in_progress",
                        "done",
                        "rejected",
                        "deferred",
                        "needs_info",
                    ],
                },
                "reason": {"type": "string"},
                "artifacts": _STRINGS,
                "gate_verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string"},
                            "verdict": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                        },
                        "required": ["gate", "verdict"],
                        "additionalProperties": False,
                    },
                },
                "gate_report": {"type": "string"},
                "decision_refs": _STRINGS,
                "impression_refs": _STRINGS,
                "questions_for_user": _STRINGS,
            },
            ["request", "status"],
        ),
        False,
    ),
    "fpga_render": (
        "Re-render pin map, floorplan, utilization, timing, waveform and "
        "gate-report PNGs from existing artifacts without rerunning tools",
        _schema(
            {
                **_CONTRACT,
                **_OUT,
                "view": {
                    "type": "string",
                    "enum": [
                        "pinmap",
                        "utilization",
                        "timing",
                        "floorplan",
                        "waveform",
                        "report",
                        "all",
                    ],
                },
            },
            ["contract_path"],
        ),
        False,
    ),
}

_MAX_IMAGES = 8
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


def tool_specs() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=description,
            inputSchema=schema,
            annotations=types.ToolAnnotations(readOnlyHint=read_only),
        )
        for name, (description, schema, read_only) in TOOLS.items()
    ]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return tool_specs()


def _str(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _contract(arguments: dict[str, object]) -> Path:
    return workspace_path(_str(arguments, "contract_path"))


def _opt_path(arguments: dict[str, object], key: str) -> Path | None:
    value = arguments.get(key)
    return workspace_path(value) if isinstance(value, str) and value else None


def _opt_str(arguments: dict[str, object], key: str) -> str | None:
    value = arguments.get(key)
    return value if isinstance(value, str) and value else None


def _strs(arguments: dict[str, object], key: str) -> list[str]:
    value = arguments.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of strings")
    items: list[str] = []
    for item in value:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, str):
            raise ValueError(f"{key} must be a list of strings")
        items.append(item)
    return items


def dispatch(name: str, arguments: dict[str, object]) -> service.Json:
    handlers: dict[str, Callable[[], service.Json]] = {
        "fpga_doctor": service.doctor_payload,
        "fpga_validate": lambda: service.validate_payload(_contract(arguments)),
        "fpga_check": lambda: service.gates_payload(
            _contract(arguments), _opt_path(arguments, "out_dir"), full=False
        ),
        "fpga_gates": lambda: service.gates_payload(
            _contract(arguments), _opt_path(arguments, "out_dir"), full=True
        ),
        "fpga_constraints": lambda: service.constraints_payload(_contract(arguments)),
        "fpga_pinmap_export": lambda: service.pinmap_payload(
            _contract(arguments), _opt_path(arguments, "out_dir")
        ),
        "fpga_regmap_export": lambda: service.regmap_payload(
            _contract(arguments), _opt_path(arguments, "out_dir")
        ),
        "fpga_lint": lambda: service.lint_payload(
            _contract(arguments), _opt_path(arguments, "out_dir")
        ),
        "fpga_sim": lambda: service.sim_payload(
            _contract(arguments), _str(arguments, "simulation"), _opt_path(arguments, "out_dir")
        ),
        "fpga_formal": lambda: service.formal_payload(
            _contract(arguments), _str(arguments, "formal"), _opt_path(arguments, "out_dir")
        ),
        "fpga_build": lambda: service.build_payload(
            _contract(arguments), _opt_path(arguments, "out_dir")
        ),
        "fpga_production_export": lambda: service.production_payload(
            _contract(arguments), _opt_path(arguments, "out_dir")
        ),
        "fpga_request": lambda: service.request_payload(
            _contract(arguments),
            _opt_path(arguments, "out_dir"),
            target=_str(arguments, "target"),
            risk=_str(arguments, "risk"),
            change=_str(arguments, "change"),
            rationale=_str(arguments, "rationale"),
            nets=_strs(arguments, "nets"),
            failing_checks=_strs(arguments, "failing_checks"),
            extra_inputs=_strs(arguments, "inputs"),
            decision_refs=_strs(arguments, "decision_refs"),
        ),
        "fpga_sim_thermal_request": lambda: service.sim_request_payload(
            _contract(arguments), _opt_path(arguments, "out_dir")
        ),
        "fpga_sim_thermal_check": lambda: service.sim_check_payload(_contract(arguments)),
        "fpga_profile": lambda: service.profile_payload(_opt_str(arguments, "profile")),
        "fpga_render": lambda: service.render_payload(
            _contract(arguments),
            _opt_path(arguments, "out_dir"),
            _opt_str(arguments, "view") or "all",
        ),
        "fpga_record_decision": lambda: record_decision(arguments),
        "fpga_record_impression": lambda: record_impression(arguments),
        "fpga_record_vision_review": lambda: record_vision_review(arguments),
        "fpga_records_status": records_summary,
        "fpga_record_reconcile": lambda: record_reconcile(arguments),
        "fpga_record_insight": lambda: record_insight(arguments),
        "fpga_record_song_receipt": lambda: record_song_receipt(arguments),
        "fpga_record_read": lambda: record_read(arguments),
        "fpga_records_digest": records_digest,
        "fpga_records_search": lambda: records_search(arguments),
        "fpga_ux_inbox": lambda: service.ux_inbox_payload(_opt_path(arguments, "workspace")),
        "fpga_ux_respond": lambda: service.ux_respond_payload(
            _opt_path(arguments, "workspace"), arguments
        ),
    }
    handler = handlers.get(name)
    if handler is None:
        return {"verdict": "fail", "detail": f"unknown tool {name}"}
    return handler()


def _image_contents(payload: dict[str, object]) -> list[types.ImageContent]:
    """Attach each produced PNG inline so a vision model sees it."""
    images = payload.get("images")
    if not isinstance(images, list):
        return []
    contents: list[types.ImageContent] = []
    notes: list[str] = []
    for value in cast(list[Any], images):
        if len(contents) >= _MAX_IMAGES:
            notes.append(f"image limit reached; {value} not embedded")
            continue
        path = Path(str(value))
        try:
            data = path.read_bytes()
        except OSError:
            notes.append(f"{value} could not be read")
            continue
        if len(data) > _MAX_IMAGE_BYTES:
            notes.append(f"{value} is {len(data)} bytes > {_MAX_IMAGE_BYTES}; not embedded")
            continue
        contents.append(
            types.ImageContent(
                type="image",
                data=base64.b64encode(data).decode("ascii"),
                mimeType="image/png",
            )
        )
    if notes:
        payload["image_notes"] = notes
    return contents


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, object]) -> types.CallToolResult:
    is_error = False
    payload: dict[str, object]
    try:
        payload = await asyncio.to_thread(dispatch, name, arguments or {})
        if name not in TOOLS:
            payload["error_type"] = "unknown_tool"
            is_error = True
    except Exception as exc:  # fail-closed transport
        payload = {
            "verdict": "fail",
            "detail": f"{name} error: {exc}",
            "error_type": type(exc).__name__,
        }
        is_error = True
    images = [] if is_error else _image_contents(payload)
    content: list[Any] = [
        types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2)),
        *images,
    ]
    return types.CallToolResult(content=content, isError=is_error)


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=f"fpga-mcp/{__version__}",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
