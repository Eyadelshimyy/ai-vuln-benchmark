"""The agent journey over the MCP gateway, against a scaffolded project:

    get_context → list models → a stamped query → validate a spec (wrong, then
    right) → render it → verify the receipt → build the report package →
    fetch the artifact back

plus the gateway's two security promises: a report name can't climb out of
``reports/``, and fetch can't read outside the output root. The tools are the
same plain functions the MCP server registers.
"""

import json

from tests.e2e.conftest import run_cli


def _spec(fact, measure="revenue"):
    return {"name": "Agent Brief", "sections": [
        {"type": "table", "title": "Revenue by region",
         "data": {"model": "sample_model", "query": {
             "fact": fact, "measures": [measure],
             "dimensions": ["dim_region.region"]}}}]}


def test_an_agent_authors_verifies_and_delivers_a_report(scaffolded):
    from tracebi import mcp_server as gw
    from tracebi.model_registry import get_model

    code, out = run_cli("run-transform", "sample_transform")
    assert code == 0, out

    context = gw.gateway_context(model="sample_model", brief=True)
    assert "sample_model" in json.dumps(context)

    models = gw.gateway_models()
    assert "sample_model" in json.dumps(models)
    info = gw.gateway_model_info("sample_model")
    fact = info["facts"][0]["name"]

    # A stamped query: the fingerprint is the model's own answer.
    stamped = gw.gateway_query("sample_model", fact, ["revenue"],
                               dimensions=["dim_region.region"])
    direct = get_model("sample_model").query(fact, ["revenue"],
                                             dimensions=["dim_region.region"])
    assert stamped["fingerprint"] == direct.fingerprint()
    assert stamped["lineage"]

    # Validate: a wrong measure is caught before anything runs, and named.
    bad = gw.gateway_validate_spec(_spec(fact, "no_such_measure"))
    assert bad["ok"] is False and "no_such_measure" in json.dumps(bad)
    good = gw.gateway_validate_spec(_spec(fact))
    assert good["ok"] is True, good

    rendered = gw.gateway_render_spec(_spec(fact), output_dir="output")
    assert rendered["ok"] is True, rendered
    verified = gw.gateway_verify_manifest(rendered["manifest_path"])
    assert verified["verdict"] == "reproduces", verified
    assert verified["summary"]["reproduces"] == len(rendered["dataset_fingerprints"])

    built = gw.gateway_build_report("sample_model/sample_dashboard")
    assert built["ok"] is True, built
    assert built["transform_contracts"], "the build joins the sink contract"
    fetched = gw.gateway_fetch_artifact(built["output_path"])
    assert "tracebi-receipt" in json.dumps(fetched)


def test_the_gateway_refuses_paths_outside_the_project(scaffolded):
    from tracebi import mcp_server as gw

    escaped = gw.gateway_build_report("../../etc/passwd")
    assert escaped["ok"] is False and ".." in escaped["errors"][0]
    fetched = gw.gateway_fetch_artifact("/etc/passwd")
    assert fetched["ok"] is False and "escapes" in fetched["errors"][0]
