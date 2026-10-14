"""Accessible, deterministic rendering for cross-engine conformance evidence."""

from __future__ import annotations

from html import escape
import json

from .cross_engine_conformance import ConformanceInventory


_CSP = (
    "default-src 'none'; base-uri 'none'; form-action 'none'; frame-src 'none'; "
    "img-src 'none'; media-src 'none'; object-src 'none'; script-src 'none'; "
    "style-src 'none'"
)
_DISCLAIMER = (
    "Numerical conformance evidence is not construct validity, fairness, or "
    "high-stakes approval. Independent-engine agreement is validation evidence, "
    "not a sole correctness oracle or a decision-authority claim."
)


def _text(value: object | None) -> str:
    """Return escaped visible text for one already-replayed manifest value."""
    if value is None:
        return "Not recorded"
    return escape(str(value), quote=True)


def _header_row(labels: tuple[str, ...]) -> str:
    """Render semantic column headers."""
    return "<tr>" + "".join(
        f'<th scope="col">{escape(label, quote=True)}</th>' for label in labels
    ) + "</tr>"


def _data_row(values: tuple[object | None, ...]) -> str:
    """Render one escaped table row."""
    return "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in values) + "</tr>"


def _key_value_table(caption: str, rows: tuple[tuple[str, object | None], ...]) -> str:
    """Render an accessible two-column key/value table."""
    rendered = ["<table>", f"<caption>{escape(caption, quote=True)}</caption>"]
    rendered.append(_header_row(("Field", "Exact value")))
    for label, value in rows:
        rendered.append(
            "<tr>"
            f'<th scope="row">{escape(label, quote=True)}</th>'
            f"<td>{_text(value)}</td>"
            "</tr>"
        )
    rendered.append("</table>")
    return "\n".join(rendered)


def _render_inventory_provenance(manifest: dict[str, object]) -> str:
    """Render immutable top-level inventory identity."""
    return _key_value_table(
        "Inventory provenance",
        (
            ("Inventory fingerprint", manifest["inventory_fingerprint"]),
            ("Package version", manifest["package_version"]),
            ("Source commit", manifest["source_commit"]),
            ("Schema version", manifest["schema_version"]),
        ),
    )


def _render_run_provenance(manifest: dict[str, object]) -> str:
    """Render exact isolated-run provenance or one explicit no-run state."""
    run = manifest["run_provenance"]
    if run is None:
        return (
            "<p>No run provenance is recorded because this inventory contains "
            "no executed evidence.</p>"
        )
    if type(run) is not dict:  # defensive; strict replay should make this unreachable
        raise ValueError("run_provenance must be a canonical dictionary")
    seeds = run["rng_seeds"]
    if type(seeds) is not list:  # defensive; strict replay should make this unreachable
        raise ValueError("rng_seeds must be a canonical list")
    seed_text = ", ".join(str(seed) for seed in seeds) if seeds else "None recorded"
    rows = (
        ("Harness commit", run["harness_commit"]),
        ("Environment kind", run["environment_kind"]),
        ("Environment SHA-256", run["environment_sha256"]),
        ("Operating system", run["operating_system"]),
        ("Architecture", run["architecture"]),
        ("RNG algorithm", run["rng_algorithm"]),
        ("RNG seeds", seed_text),
        ("Mapping schema version", run["mapping_schema_version"]),
        ("Mapping SHA-256", run["mapping_sha256"]),
        ("Model configuration SHA-256", run["model_configuration_sha256"]),
        ("Convergence controls SHA-256", run["convergence_controls_sha256"]),
        ("Tolerance SHA-256", run["tolerance_sha256"]),
        ("Tolerance rationale", run["tolerance_rationale"]),
        ("Raw output SHA-256", run["raw_output_sha256"]),
        ("Normalized output SHA-256", run["normalized_output_sha256"]),
        ("License classification", run["license_classification"]),
        ("Redistribution status", run["redistribution_status"]),
    )
    return _key_value_table("Isolated run provenance", rows)


def _render_capabilities(manifest: dict[str, object]) -> str:
    """Render one exact row per advertised capability."""
    capabilities = manifest["capabilities"]
    if type(capabilities) is not list:  # defensive; strict replay should make this unreachable
        raise ValueError("capabilities must be a canonical list")
    labels = (
        "Capability",
        "Public entry point",
        "Coverage status",
        "Estimand",
        "Likelihood family",
        "Parameterization",
        "Identification",
        "Comparison scope",
        "Evidence rows",
    )
    rows = ["<table>", "<caption>Capability coverage</caption>", _header_row(labels)]
    for capability in capabilities:
        if type(capability) is not dict:
            raise ValueError("capability must be a canonical dictionary")
        evidence = capability["evidence"]
        if type(evidence) is not list:
            raise ValueError("capability evidence must be a canonical list")
        rows.append(
            _data_row(
                (
                    capability["capability_id"],
                    capability["public_entrypoint"],
                    capability["coverage_status"],
                    capability["estimand"],
                    capability["likelihood_family"],
                    capability["parameterization"],
                    capability["identification"],
                    capability["comparison_scope"],
                    len(evidence),
                )
            )
        )
    rows.append("</table>")
    return "\n".join(rows)


def _render_evidence(manifest: dict[str, object]) -> str:
    """Render long-form capability × engine evidence with explicit empty states."""
    capabilities = manifest["capabilities"]
    if type(capabilities) is not list:
        raise ValueError("capabilities must be a canonical list")
    labels = (
        "Capability",
        "Coverage status",
        "Evidence id",
        "Engine",
        "Engine version",
        "Engine source",
        "Engine license",
        "Layer",
        "Execution status",
        "Mapping version",
        "Mapping SHA-256",
        "Fixture SHA-256",
        "Environment SHA-256",
        "Artifact SHA-256",
        "Limitation",
    )
    rows = [
        "<table>",
        "<caption>Capability × engine conformance evidence</caption>",
        _header_row(labels),
    ]
    evidence_count = 0
    for capability in capabilities:
        if type(capability) is not dict:
            raise ValueError("capability must be a canonical dictionary")
        evidence_values = capability["evidence"]
        if type(evidence_values) is not list:
            raise ValueError("capability evidence must be a canonical list")
        if not evidence_values:
            rows.append(
                _data_row(
                    (
                        capability["capability_id"],
                        capability["coverage_status"],
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "not_executed",
                        None,
                        None,
                        None,
                        None,
                        None,
                        "No independent engine evidence row is recorded for this capability.",
                    )
                )
            )
            continue
        for evidence in evidence_values:
            if type(evidence) is not dict:
                raise ValueError("evidence must be a canonical dictionary")
            engine = evidence["engine"]
            if type(engine) is not dict:
                raise ValueError("engine must be a canonical dictionary")
            evidence_count += 1
            rows.append(
                _data_row(
                    (
                        capability["capability_id"],
                        capability["coverage_status"],
                        evidence["evidence_id"],
                        engine["engine_id"],
                        engine["engine_version"],
                        engine["source_reference"],
                        engine["license_classification"],
                        evidence["layer"],
                        evidence["execution_status"],
                        evidence["parameter_mapping_version"],
                        evidence["parameter_mapping_sha256"],
                        evidence["fixture_sha256"],
                        evidence["environment_sha256"],
                        evidence["artifact_sha256"],
                        evidence["limitation"],
                    )
                )
            )
    rows.append("</table>")
    if evidence_count == 0:
        rows.insert(
            0,
            "<p>No independent engine evidence rows are recorded for this inventory.</p>",
        )
    return "\n".join(rows)


def _long_form_rows(manifest: dict[str, object]) -> list[dict[str, object | None]]:
    """Flatten canonical capability/evidence records without numerical reinterpretation."""
    capabilities = manifest["capabilities"]
    if type(capabilities) is not list:  # defensive; strict replay should make this unreachable
        raise ValueError("capabilities must be a canonical list")
    provenance = {
        "inventory_fingerprint": manifest["inventory_fingerprint"],
        "package_version": manifest["package_version"],
        "source_commit": manifest["source_commit"],
        "schema_version": manifest["schema_version"],
    }
    rows: list[dict[str, object | None]] = []
    for capability in capabilities:
        if type(capability) is not dict:
            raise ValueError("capability must be a canonical dictionary")
        capability_fields = {
            "capability_id": capability["capability_id"],
            "public_entrypoint": capability["public_entrypoint"],
            "coverage_status": capability["coverage_status"],
            "estimand": capability["estimand"],
            "likelihood_family": capability["likelihood_family"],
            "parameterization": capability["parameterization"],
            "identification": capability["identification"],
            "comparison_scope": capability["comparison_scope"],
        }
        evidence_values = capability["evidence"]
        if type(evidence_values) is not list:
            raise ValueError("capability evidence must be a canonical list")
        if not evidence_values:
            rows.append(
                {
                    **provenance,
                    **capability_fields,
                    "evidence_id": None,
                    "engine_id": None,
                    "engine_version": None,
                    "engine_source_reference": None,
                    "engine_license_classification": None,
                    "layer": None,
                    "execution_status": "not_executed",
                    "parameter_mapping_version": None,
                    "parameter_mapping_sha256": None,
                    "fixture_sha256": None,
                    "environment_sha256": None,
                    "artifact_sha256": None,
                    "limitation": "No independent engine evidence row is recorded for this capability.",
                }
            )
            continue
        for evidence in evidence_values:
            if type(evidence) is not dict:
                raise ValueError("evidence must be a canonical dictionary")
            engine = evidence["engine"]
            if type(engine) is not dict:
                raise ValueError("engine must be a canonical dictionary")
            rows.append(
                {
                    **provenance,
                    **capability_fields,
                    "evidence_id": evidence["evidence_id"],
                    "engine_id": engine["engine_id"],
                    "engine_version": engine["engine_version"],
                    "engine_source_reference": engine["source_reference"],
                    "engine_license_classification": engine["license_classification"],
                    "layer": evidence["layer"],
                    "execution_status": evidence["execution_status"],
                    "parameter_mapping_version": evidence["parameter_mapping_version"],
                    "parameter_mapping_sha256": evidence["parameter_mapping_sha256"],
                    "fixture_sha256": evidence["fixture_sha256"],
                    "environment_sha256": evidence["environment_sha256"],
                    "artifact_sha256": evidence["artifact_sha256"],
                    "limitation": evidence["limitation"],
                }
            )
    return rows


def render_conformance_long_form_json(manifest_json: str) -> str:
    """Render downloadable long-form capability × engine evidence as strict JSON.

    Input is replayed through :meth:`ConformanceInventory.from_json` before any
    projection. The output is a deterministic, provenance-bound table encoded
    as JSON; it performs no comparison, alignment, scoring, or statistical
    calculation and preserves explicit non-executed/no-engine states.
    """
    inventory = ConformanceInventory.from_json(manifest_json)
    rows = _long_form_rows(inventory.to_manifest())
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def render_conformance_report(manifest_json: str) -> tuple[str, str]:
    """Render strict conformance JSON as standalone accessible HTML plus JSON.

    Parsing and integrity validation are delegated to
    :meth:`ConformanceInventory.from_json`. The renderer performs no numerical
    comparison, discrepancy, likelihood, scoring, alignment, or uncertainty
    calculation; it only projects already-validated source-free evidence.
    """
    inventory = ConformanceInventory.from_json(manifest_json)
    manifest = inventory.to_manifest()
    canonical_json = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    body = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f'<meta http-equiv="Content-Security-Policy" content="{escape(_CSP, quote=True)}">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Cross-engine conformance evidence</title>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>Cross-engine conformance evidence</h1>",
        f"<p>{escape(_DISCLAIMER, quote=True)}</p>",
        "<p>Exact values are shown in text; this report has no hover-only evidence.</p>",
        "<section aria-labelledby=\"inventory-provenance\">",
        '<h2 id="inventory-provenance">Inventory provenance</h2>',
        _render_inventory_provenance(manifest),
        "</section>",
        "<section aria-labelledby=\"run-provenance\">",
        '<h2 id="run-provenance">Run provenance</h2>',
        _render_run_provenance(manifest),
        "</section>",
        "<section aria-labelledby=\"capability-coverage\">",
        '<h2 id="capability-coverage">Capability coverage</h2>',
        _render_capabilities(manifest),
        "</section>",
        "<section aria-labelledby=\"engine-evidence\">",
        '<h2 id="engine-evidence">Capability × engine evidence</h2>',
        _render_evidence(manifest),
        "</section>",
        "</main>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(body), canonical_json


__all__ = ["render_conformance_long_form_json", "render_conformance_report"]
