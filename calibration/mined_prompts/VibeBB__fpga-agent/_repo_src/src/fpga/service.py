"""Entry points shared by the CLI and the MCP server. Each returns a JSON
payload with a fail-closed ``verdict``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from . import doctor, flow, liaison, program
from .contract import FpgaContract, load_contract, resolve
from .devices import DeviceProfile, bundled_ids, load_profile
from .formal import run_formal
from .gates import (
    FAIL,
    PASS,
    Check,
    implementation_checks,
    run_gates,
    write_envelope_v3,
    write_outputs,
)
from .interchange import sha256_file
from .projections import constraints_text, pinmap_export, pinmap_markdown, write_text
from .regmap import hdl_package_text, regmap_export
from .requests import write_request
from .sim import run_simulation
from .sim_thermal import thermal_check, write_sim_request
from .toolrun import run_tool
from .workspace import workspace_path, workspace_root

type Json = dict[str, object]


def default_out(contract_path: Path) -> Path:
    return contract_path.resolve().parent / "fpga-reports"


def _load(contract_path: Path) -> tuple[FpgaContract, DeviceProfile]:
    contract = load_contract(contract_path)
    return contract, load_profile(contract.device.profile, [contract_path.resolve().parent])


def _circuit_sha(contract: FpgaContract, contract_path: Path) -> str | None:
    """sha256 of the circuit connectivity file the contract links, if any."""
    if contract.circuit is None:
        return None
    path = resolve(contract_path.resolve(), contract.circuit.connectivity)
    return sha256_file(path) if path.is_file() else None


def _sources(contract: FpgaContract, contract_path: Path) -> list[Path]:
    """The contract plus the circuit connectivity it links, when present."""
    sources = [contract_path]
    if contract.circuit is not None:
        path = resolve(contract_path.resolve(), contract.circuit.connectivity)
        if path.is_file():
            sources.append(path)
    return sources


def _envelope_v3(artifact: Path, sources: list[Path], stage: str) -> Json | None:
    """Write the envelope v3 sidecar; a refusal becomes the stage's fail payload."""
    try:
        write_envelope_v3(artifact, sources)
    except ValueError as exc:
        return {"verdict": FAIL, "stage": stage, "detail": f"provenance refused: {exc}"}
    return None


def doctor_payload() -> Json:
    results = doctor.checks()
    ok = all(item.status != "fail" for item in results)
    return {
        "verdict": PASS if ok else FAIL,
        "checks": [item.model_dump(mode="json") for item in results],
    }


def validate_payload(contract_path: Path) -> Json:
    try:
        contract, profile = _load(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "validate", "detail": str(exc)}
    return {
        "verdict": PASS,
        "stage": "validate",
        "design": contract.name,
        "language": contract.design_language,
        "profile": profile.id,
        "part": profile.part,
        "pins": len(contract.pins),
        "libraries": [lib.name for lib in contract.libraries],
        "simulations": [s.id for s in contract.simulations],
        "formal": [f.id for f in contract.formal],
    }


def _images(written: list[Path]) -> list[str]:
    return [str(p) for p in written if p.suffix == ".png" and p.is_file()]


_VISION_CHECKLIST_TOKENS: tuple[tuple[str, str], ...] = (
    (".fpga-pinmap", "pinmap"),
    (".fpga-floorplan", "floorplan"),
    (".fpga-utilization", "utilization"),
    (".fpga-timing", "timing"),
    (".fpga-wave", "waveform"),
    (".fpga-report", "report"),
)


def _checklist_slug(image: str) -> str:
    """Best-effort checklist slug for a report raster, by file name."""
    stem = Path(image).stem.lower()
    for token, checklist in _VISION_CHECKLIST_TOKENS:
        if token in stem:
            return checklist
    return "render"


def vision_points(images: list[str]) -> list[dict[str, str]]:
    """Vision-review packet for the renders a payload already lists."""
    return [
        {
            "image_path": image,
            "checklist": _checklist_slug(image),
            "record_with": "fpga_record_vision_review",
        }
        for image in images
    ]


def gates_payload(contract_path: Path, out_dir: Path | None, *, full: bool) -> Json:
    out = out_dir or default_out(contract_path)
    report = run_gates(contract_path, out, full=full)
    written = write_outputs(contract_path.resolve(), report, out)
    payload: Json = json.loads(report.model_dump_json())
    payload["written"] = [str(p) for p in written]
    images = _images(written)
    if images:
        payload["images"] = images
        payload["vision_points"] = vision_points(images)
    return payload


def constraints_payload(contract_path: Path) -> Json:
    try:
        contract, profile = _load(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "constraints", "detail": str(exc)}
    path = write_text(
        resolve(contract_path.resolve(), contract.build.constraints),
        constraints_text(contract, profile),
    )
    return {"verdict": PASS, "stage": "constraints", "written": [str(path)]}


def pinmap_payload(contract_path: Path, out_dir: Path | None) -> Json:
    try:
        contract, profile = _load(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "pinmap", "detail": str(exc)}
    out = out_dir or default_out(contract_path)
    pinmap = pinmap_export(
        contract, profile, sha256_file(contract_path), _circuit_sha(contract, contract_path)
    )
    json_path = write_text(
        out / f"{contract.name}.fpga-pinmap.json",
        json.dumps(pinmap.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )
    refused = _envelope_v3(json_path, _sources(contract, contract_path), "pinmap")
    if refused is not None:
        return refused
    md_path = write_text(out / f"{contract.name}.fpga-pinmap.md", pinmap_markdown(pinmap))
    written = [json_path, md_path]
    try:
        from .render import pinmap_canvas, write_png

        png_path = out / f"{contract.name}.fpga-pinmap.png"
        write_png(pinmap_canvas(contract, profile), png_path)
        written.append(png_path)
    except Exception as exc:  # a render failure never fails the export
        return {
            "verdict": PASS,
            "stage": "pinmap",
            "written": [str(p) for p in written],
            "render_errors": [str(exc)],
        }
    payload: Json = {"verdict": PASS, "stage": "pinmap", "written": [str(p) for p in written]}
    images = _images(written)
    payload["images"] = images
    if images:
        payload["vision_points"] = vision_points(images)
    return payload


def regmap_payload(contract_path: Path, out_dir: Path | None) -> Json:
    """Write the HDL register constants and ``<name>.fpga-regmap.json`` for firmware-agent."""
    try:
        contract = load_contract(contract_path)
        if contract.registers is None:
            raise ValueError("contract declares no registers")
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "regmap", "detail": str(exc)}
    out = out_dir or default_out(contract_path)
    package = write_text(
        resolve(contract_path.resolve(), contract.registers.hdl_package),
        hdl_package_text(contract),
    )
    export = regmap_export(contract, sha256_file(contract_path))
    path = write_text(
        out / f"{contract.name}.fpga-regmap.json",
        json.dumps(export.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )
    refused = _envelope_v3(path, [contract_path], "regmap")
    if refused is not None:
        return refused
    return {
        "verdict": PASS,
        "stage": "regmap",
        "bus": export.bus,
        "registers": len(export.registers),
        "written": [str(package), str(path)],
    }


def sim_payload(contract_path: Path, sim_id: str, out_dir: Path | None) -> Json:
    try:
        contract = load_contract(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "sim", "detail": str(exc)}
    sim = contract.simulation(sim_id)
    if sim is None:
        return {"verdict": FAIL, "stage": "sim", "detail": f"unknown simulation {sim_id}"}
    out = out_dir or default_out(contract_path)
    result = run_simulation(contract, contract_path.resolve(), sim, out)
    payload: Json = {
        "verdict": PASS if result.ok else FAIL,
        "stage": "sim",
        "simulation": sim.id,
        "detail": result.detail,
        "argv": result.argv,
        "matched": result.matched,
        "missing": result.missing,
        "forbidden": result.forbidden,
        "transcript": str(result.transcript) if result.transcript else None,
        "waveform": str(result.waveform) if result.waveform else None,
    }
    if result.waveform is not None:
        try:
            from .render import waveform_canvas, write_png

            png_path = out / f"{contract.name}.fpga-wave-{sim.id}.png"
            write_png(waveform_canvas(result.waveform, f"{contract.name} wave {sim.id}"), png_path)
            images = [str(png_path)]
            payload["images"] = images
            payload["vision_points"] = vision_points(images)
        except Exception as exc:
            payload["render_errors"] = [str(exc)]
    return payload


def formal_payload(contract_path: Path, run_id: str, out_dir: Path | None) -> Json:
    try:
        contract = load_contract(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "formal", "detail": str(exc)}
    run = contract.formal_run(run_id)
    if run is None:
        return {"verdict": FAIL, "stage": "formal", "detail": f"unknown formal run {run_id}"}
    out = out_dir or default_out(contract_path)
    result = run_formal(contract, contract_path.resolve(), run, out)
    return {
        "verdict": PASS if result.ok else FAIL,
        "stage": "formal",
        "formal": run.id,
        "status": result.status,
        "detail": result.detail,
        "argv": result.argv,
        "transcript": str(out / f"formal-{run.id}.log"),
    }


def lint_payload(contract_path: Path, out_dir: Path | None) -> Json:
    try:
        contract = load_contract(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "lint", "detail": str(exc)}
    out = out_dir or default_out(contract_path)
    run = flow.lint(contract, contract_path.resolve(), out)
    return {
        "verdict": PASS if run.ok else FAIL,
        "stage": "lint",
        "detail": run.detail,
        "argv": run.argv,
        "evidence": run.evidence,
        "transcript": str(out / "lint.log"),
    }


def build_payload(contract_path: Path, out_dir: Path | None) -> Json:
    """Synthesis, place and route and packing, without simulation or formal."""
    try:
        contract, profile = _load(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "build", "detail": str(exc)}
    out = out_dir or default_out(contract_path)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = implementation_checks(contract, contract_path.resolve(), profile, out)
    ok = all(c.status != "fail" for c in checks)
    payload: Json = {
        "verdict": PASS if ok else FAIL,
        "stage": "build",
        "checks": [c.model_dump(mode="json") for c in checks],
        "note": "advisory; `fpga gates` is the authoritative run",
    }
    from .render import render_view_pngs

    rendered, skipped = render_view_pngs(
        contract,
        profile,
        out,
        build_dir=resolve(contract_path.resolve(), contract.build.dir),
        views=["pinmap", "utilization", "timing", "floorplan", "waveform"],
    )
    if rendered:
        payload["renders"] = rendered
        images = [str(item["path"]) for item in rendered]
        payload["images"] = images
        payload["vision_points"] = vision_points(images)
    if skipped:
        payload["render_errors"] = skipped
    return payload


def program_payload(
    contract_path: Path, out_dir: Path | None, confirm_sha256: str, *, dry_run: bool
) -> Json:
    try:
        contract = load_contract(contract_path)
        planned = program.plan(
            contract, contract_path.resolve(), out_dir or default_out(contract_path), confirm_sha256
        )
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "program", "detail": str(exc)}
    if dry_run:
        return {"verdict": PASS, "stage": "program", "dry_run": True, "argv": planned.argv}
    run = run_tool(planned.argv, contract_path.resolve().parent, None, 600)
    return {
        "verdict": PASS if run.ok else FAIL,
        "stage": "program",
        "argv": planned.argv,
        "sha256": planned.sha256,
        "detail": run.detail,
        "output": run.output[-4000:],
    }


def request_payload(
    contract_path: Path,
    out_dir: Path | None,
    *,
    target: str,
    risk: str,
    change: str,
    rationale: str,
    nets: list[str],
    failing_checks: list[str],
    extra_inputs: list[str] | None = None,
    decision_refs: list[str] | None = None,
) -> Json:
    try:
        contract = load_contract(contract_path)
        request, path = write_request(
            contract_path,
            contract.name,
            out_dir or default_out(contract_path),
            target=target,
            risk=risk,
            change=change,
            rationale=rationale,
            nets=nets,
            failing_checks=failing_checks,
            extra_inputs=extra_inputs,
            decision_refs=decision_refs,
        )
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "request", "detail": str(exc)}
    return {"verdict": PASS, "stage": "request", "id": request.id, "written": [str(path)]}


def sim_request_payload(contract_path: Path, out_dir: Path | None) -> Json:
    """Write the simulation-agent thermal brief and request for the FPGA package."""
    try:
        contract = load_contract(contract_path)
        if contract.thermal is None:
            raise ValueError("contract has no thermal section")
        root = workspace_root()
        out = workspace_path(out_dir or default_out(contract_path), root=root)
        return write_sim_request(contract, out, root=root)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "sim-request", "detail": str(exc)}


def sim_check_payload(contract_path: Path) -> Json:
    """Read simulation-agent's hash-bound thermal answer (pass, fail or unknown)."""
    try:
        contract = load_contract(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "sim-check", "detail": str(exc)}
    return thermal_check(contract, contract_path, workspace_root())


def production_payload(contract_path: Path, out_dir: Path | None) -> Json:
    """Write ``<name>.fpga-production.json`` for production-engineering-agent."""
    try:
        contract, profile = _load(contract_path)
        out = out_dir or default_out(contract_path)
        artifact = program.production_export(contract, contract_path, profile, out)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "production", "detail": str(exc)}
    path = out / f"{contract.name}.fpga-production.json"
    write_text(path, artifact.model_dump_json(indent=2) + "\n")
    refused = _envelope_v3(path, [contract_path], "production")
    if refused is not None:
        return refused
    return {
        "verdict": PASS,
        "stage": "production",
        "bitstream_sha256": artifact.bitstream_sha256,
        "target": artifact.target,
        "written": [str(path)],
    }


def render_payload(contract_path: Path, out_dir: Path | None, view: str) -> Json:
    """Re-render the report images from existing artifacts without rerunning tools."""
    try:
        contract, profile = _load(contract_path)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "render", "detail": str(exc)}
    out = out_dir or default_out(contract_path)
    views = None if view == "all" else [view]
    from .render import render_view_pngs

    try:
        rendered, skipped = render_view_pngs(
            contract,
            profile,
            out,
            build_dir=resolve(contract_path.resolve(), contract.build.dir),
            views=views,
        )
    except Exception as exc:
        return {"verdict": FAIL, "stage": "render", "detail": str(exc)}
    return {
        "verdict": PASS,
        "stage": "render",
        "rendered": rendered,
        "skipped": skipped,
        "images": (images := [str(item["path"]) for item in rendered]),
        "vision_points": vision_points(images),
    }


def ux_inbox_payload(workspace: Path | None) -> Json:
    return cast(Json, liaison.inbox(workspace))


def ux_respond_payload(workspace: Path | None, payload: dict[str, object]) -> Json:
    """Validate and write a UX response; refusals come back as a fail verdict."""
    request = str(payload.get("request", ""))
    status = str(payload.get("status", ""))
    try:
        result = liaison.respond(
            workspace,
            request,
            status,
            reason=str(payload.get("reason", "")),
            artifacts=[str(p) for p in cast(list[object], payload.get("artifacts") or [])],
            gate_verdicts=[
                cast(dict[str, str], v)
                for v in cast(list[object], payload.get("gate_verdicts") or [])
            ],
            gate_report=cast(str | Path | None, payload.get("gate_report")),
            decision_refs=[str(r) for r in cast(list[object], payload.get("decision_refs") or [])],
            impression_refs=[
                str(r) for r in cast(list[object], payload.get("impression_refs") or [])
            ],
            questions_for_user=[
                str(q) for q in cast(list[object], payload.get("questions_for_user") or [])
            ],
        )
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "ux-respond", "detail": str(exc)}
    return cast(Json, result)


def profile_payload(profile_id: str | None) -> Json:
    if profile_id is None:
        return {"verdict": PASS, "stage": "profile", "bundled": bundled_ids()}
    try:
        profile = load_profile(profile_id)
    except (OSError, ValueError, ValidationError) as exc:
        return {"verdict": FAIL, "stage": "profile", "detail": str(exc)}
    payload: Json = json.loads(profile.model_dump_json())
    payload["verdict"] = PASS
    return payload
