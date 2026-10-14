"""fpga command line interface.

Subcommands:
  doctor       probe the toolchain
  validate     validate a <name>.fpga.json contract against its device profile
  check        static gates (contract, provenance, pins, constraints, netlist match)
  gates        every gate: static + lint, simulation, formal, synthesis, P&R,
               timing, utilization, bitstream
  constraints  regenerate the contract's PCF/LPF/CST constraint file
  pinmap       export <name>.fpga-pinmap.json for electrical-circuit-agent
  regmap       write the HDL register constants and <name>.fpga-regmap.json
               for firmware-agent
  lint         analyse and elaborate the design top (NVC or Verilator)
  sim          run one declared simulation
  formal       run one declared SymbiYosys proof
  build        synthesis, place and route, bitstream (advisory)
  program      host-only: program a board with openFPGALoader
  production   export <name>.fpga-production.json (gated bitstream + loader
               options) for production-engineering-agent
  request      write a change request to a sibling agent
  sim-request  write the FPGA package thermal brief + request for simulation-agent
  sim-check    read simulation-agent's hash-bound thermal answer
  profile      list bundled device profiles or print one
  record       append a VibeBB Record Protocol v2 record (decision, impression,
               vision-review, reconcile, insight, song-receipt, read) or print
               the records status, digest or search results
  render       re-render report PNGs from existing artifacts (no tool runs)
  ux inbox     triage liaison requests UX-creator addressed to this agent
  ux respond   write liaison/<id>.ux-response.json (--json carries the fields)

Every command prints a JSON payload; exit 0 only when verdict is pass.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from . import service


def _emit(payload: service.Json) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("verdict") == "pass" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpga")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--warn", action="store_true", help="always exit 0")
    for name in ("validate", "constraints"):
        sub.add_parser(name).add_argument("contract", type=Path)
    for name in ("check", "gates", "pinmap", "regmap", "lint", "build", "production"):
        command = sub.add_parser(name)
        command.add_argument("contract", type=Path)
        command.add_argument("--out", type=Path)
    for name in ("sim", "formal"):
        command = sub.add_parser(name)
        command.add_argument("contract", type=Path)
        command.add_argument("--id", required=True)
        command.add_argument("--out", type=Path)
    prog = sub.add_parser("program")
    prog.add_argument("contract", type=Path)
    prog.add_argument("--confirm-sha256", required=True)
    prog.add_argument("--dry-run", action="store_true")
    prog.add_argument("--out", type=Path)
    request = sub.add_parser("request")
    request.add_argument("contract", type=Path)
    request.add_argument("--target", required=True)
    request.add_argument("--risk", choices=("low", "high"), required=True)
    request.add_argument("--change", required=True)
    request.add_argument("--rationale", required=True)
    request.add_argument("--net", dest="nets", action="append", default=[])
    request.add_argument("--failing-check", dest="failing_checks", action="append", default=[])
    request.add_argument("--input", dest="extra_inputs", action="append", default=[])
    request.add_argument(
        "--decision-ref",
        dest="decision_refs",
        action="append",
        required=True,
        help="event_id of a record in observations/fpga/decisions.jsonl",
    )
    request.add_argument("--out", type=Path)
    sim_request = sub.add_parser("sim-request")
    sim_request.add_argument("contract", type=Path)
    sim_request.add_argument("--out", type=Path)
    sub.add_parser("sim-check").add_argument("contract", type=Path)
    sub.add_parser("profile").add_argument("id", nargs="?")
    record = sub.add_parser("record", help="append a VibeBB Record Protocol record")
    record.add_argument(
        "kind",
        choices=(
            "decision",
            "impression",
            "vision-review",
            "reconcile",
            "insight",
            "song-receipt",
            "read",
            "status",
            "digest",
            "search",
        ),
    )
    record.add_argument("--json", default=None, help="JSON object file with the record fields")
    render = sub.add_parser("render", help="re-render report PNGs from existing artifacts")
    render.add_argument("contract", type=Path)
    render.add_argument(
        "--view",
        choices=("pinmap", "utilization", "timing", "floorplan", "waveform", "report", "all"),
        default="all",
    )
    render.add_argument("--out", type=Path)
    ux = sub.add_parser("ux", help="SLP v2 liaison with UX-creator")
    ux_sub = ux.add_subparsers(dest="ux_command", required=True)
    ux_sub.add_parser("inbox").add_argument("--workspace", type=Path, default=None)
    respond = ux_sub.add_parser("respond")
    respond.add_argument("--workspace", type=Path, default=None)
    respond.add_argument("--json", required=True, help="JSON file with the response fields")
    return parser


def _cmd_record(kind: str, json_arg: str | None) -> int:
    from .records import RECORDERS, records_digest, records_search, records_summary

    if kind == "status":
        return _emit(records_summary())
    if kind == "digest":
        return _emit(records_digest())
    if kind == "search" and not json_arg:
        return _emit(records_search({}))
    if not json_arg:
        raise SystemExit(f"record {kind} requires --json")
    try:
        payload: object = json.loads(Path(json_arg).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("record JSON must be an object")
        if kind == "search":
            return _emit(records_search(cast(Mapping[str, Any], payload)))
        return _emit(RECORDERS[kind](cast(Mapping[str, Any], payload)))
    except (OSError, ValueError) as exc:
        return _emit({"verdict": "fail", "stage": "record", "detail": str(exc)})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command: str = args.command
    if command == "doctor":
        code = _emit(service.doctor_payload())
        return 0 if args.warn else code
    if command == "validate":
        return _emit(service.validate_payload(args.contract))
    if command in ("check", "gates"):
        return _emit(service.gates_payload(args.contract, args.out, full=command == "gates"))
    if command == "constraints":
        return _emit(service.constraints_payload(args.contract))
    if command == "pinmap":
        return _emit(service.pinmap_payload(args.contract, args.out))
    if command == "regmap":
        return _emit(service.regmap_payload(args.contract, args.out))
    if command == "lint":
        return _emit(service.lint_payload(args.contract, args.out))
    if command == "build":
        return _emit(service.build_payload(args.contract, args.out))
    if command == "sim":
        return _emit(service.sim_payload(args.contract, args.id, args.out))
    if command == "formal":
        return _emit(service.formal_payload(args.contract, args.id, args.out))
    if command == "sim-request":
        return _emit(service.sim_request_payload(args.contract, args.out))
    if command == "sim-check":
        return _emit(service.sim_check_payload(args.contract))
    if command == "production":
        return _emit(service.production_payload(args.contract, args.out))
    if command == "program":
        return _emit(
            service.program_payload(
                args.contract, args.out, args.confirm_sha256, dry_run=args.dry_run
            )
        )
    if command == "record":
        return _cmd_record(args.kind, args.json)
    if command == "render":
        return _emit(service.render_payload(args.contract, args.out, args.view))
    if command == "ux":
        if args.ux_command == "inbox":
            return _emit(service.ux_inbox_payload(args.workspace))
        try:
            fields: object = json.loads(Path(args.json).read_text(encoding="utf-8"))
            if not isinstance(fields, dict):
                raise ValueError("ux respond --json must hold a JSON object")
            return _emit(
                service.ux_respond_payload(args.workspace, cast("dict[str, object]", fields))
            )
        except (OSError, ValueError) as exc:
            return _emit({"verdict": "fail", "stage": "ux-respond", "detail": str(exc)})
    if command == "request":
        return _emit(
            service.request_payload(
                args.contract,
                args.out,
                target=args.target,
                risk=args.risk,
                change=args.change,
                rationale=args.rationale,
                nets=args.nets,
                failing_checks=args.failing_checks,
                extra_inputs=args.extra_inputs,
                decision_refs=args.decision_refs,
            )
        )
    return _emit(service.profile_payload(args.id))


if __name__ == "__main__":
    raise SystemExit(main())
