"""`womblex plan`: what a workflow would run and spend, offline, the shard directory untouched."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_workflow_runner import _PRESET, _resolve, _stored, shards  # noqa: F401
from womblex.analyse.embed_stage import plan_embedding
from womblex.analyse.enrich_stage import _load_narratives
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.workflow_plan import plan_workflow
from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema
from womblex.config import EnrichmentConfig
from womblex.process.chunk_stage import _batch_bases
from womblex.utils.token_packer import TokenCounter

_NORMALISED = {"processing": {"text_source": "normalised"}}
_STEPS = ["normalise", "chunk", "enrich", "embed"]


@pytest.fixture(autouse=True)
def _no_paid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("plan reached a paid or model-probing path")

    monkeypatch.setattr("womblex.utils.isaacus_client.make_isaacus_client", boom)
    monkeypatch.setattr("womblex.utils.model_check.check_models", boom)


def _snapshot(d: Path) -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in d.iterdir()}


def test_counts_are_the_stages_own_and_the_directory_is_untouched(shards: Path) -> None:  # noqa: F811
    before = _snapshot(shards)
    plan = plan_workflow(_stored(_STEPS, config=_NORMALISED), shards, resolve_preset=_resolve)
    assert _snapshot(shards) == before  # no normalised overlay, no chunks written
    assert [s.action for s in plan.steps] == ["run"] * 4
    enrich, embed = plan.steps[2].tokens, plan.steps[3].tokens
    assert enrich is not None and embed is not None
    assert enrich.basis == embed.basis == "simulated" and enrich.tokens > 0 and embed.tokens > 0

    counter = TokenCounter(EnrichmentConfig().tokenizer)  # the oracle: the same text counted directly
    narrative, _ = _load_narratives(_batch_bases(shards)[0], "elements")  # normalise is a passthrough here
    assert enrich.detail["narrative_tokens"] == sum(counter.count_batch(list(narrative.values())))


def test_embed_count_equals_a_real_chunk_run(shards: Path, tmp_path: Path) -> None:  # noqa: F811
    from womblex.cloud.stage_contracts import RunContext
    from womblex.cloud.workflow_runner import resolve_workflow_config

    stored = _stored(["chunk", "embed"])
    plan = plan_workflow(stored, shards, resolve_preset=_resolve)
    STAGE_CONTRACTS["chunk"].run(shards, resolve_workflow_config(stored, _resolve), RunContext())
    real = plan_embedding(shards)
    got = plan.steps[1].tokens
    assert got is not None and (got.tokens, got.requests, got.texts) == (real.tokens, real.requests, real.chunks)


def test_a_step_that_cannot_run_offline_leaves_the_count_pending(
    shards: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.setitem(STAGE_CONTRACTS, "chunk", replace(
        STAGE_CONTRACTS["chunk"], needs_isaacus_api_for=lambda _c: True))
    plan = plan_workflow(_stored(["chunk", "embed"]), shards, resolve_preset=_resolve, prices={"embed": 2.0})
    t = plan.steps[1].tokens
    assert t is not None and t.basis == "pending" and "'chunk'" in t.note and t.tokens == 0
    assert t.assumed_cost is None  # no count, so no $0.00


def test_price_appears_only_when_assumed(shards: Path) -> None:  # noqa: F811
    stored = _stored(["chunk", "embed"])
    assert plan_workflow(stored, shards, resolve_preset=_resolve).steps[1].tokens.assumed_cost is None  # type: ignore[union-attr]
    priced = plan_workflow(stored, shards, resolve_preset=_resolve, prices={"embed": 2.0}).steps[1].tokens
    assert priced is not None and priced.assumed_cost == pytest.approx(priced.tokens / 1e6 * 2.0)
    assert priced.assumed_price_per_million == 2.0


def test_cli_json_prices_and_exit_codes(shards: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    from womblex.cli.workflow import cmd_plan
    from womblex.ui.presets import build_preset_record, preset_filename, serialise_preset_record

    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    with WorkflowStore(dsn) as store:
        store.save("w", _stored(["chunk", "embed"]).spec)
    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp_path / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))

    def args(**kw) -> argparse.Namespace:
        base = {"workflow_id": "w", "shards": shards, "dsn": dsn, "presets_dir": tmp_path,
                "models_check": None, "assume_price": [], "json": True}
        return argparse.Namespace(**{**base, **kw})

    assert cmd_plan(args()) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["steps"][1]["tokens"]["basis"] == "simulated"
    assert cmd_plan(args(json=False, assume_price=["embed=3"])) == 0
    assert "ASSUMPTION" in capsys.readouterr().out
    assert cmd_plan(args(assume_price=["chunk=1"])) == 1
    with WorkflowStore(dsn) as store:
        store.save("refused", _stored(["embed"]).spec)  # chunks are not on disk
    for wid, code in (("absent", "workflow_unavailable"), ("refused", "step_order")):  # still one JSON object
        capsys.readouterr()
        assert cmd_plan(args(workflow_id=wid)) == 1 and json.loads(capsys.readouterr().out)["code"] == code


def test_layout_skips_only_when_every_sidecar_carries_the_fingerprint(shards: Path) -> None:  # noqa: F811
    from womblex.cloud.workflow_runner import resolve_workflow_config
    from womblex.store.layout_output import layout_fingerprint, write_layout_regions

    stored = _stored(["layout"])
    fp = layout_fingerprint(resolve_workflow_config(stored, _resolve))
    for dpi, action in ((fp.dpi + 1, "run"), (fp.dpi, "skip")):
        for base in _batch_bases(shards):
            write_layout_regions([], base, replace(fp, dpi=dpi), redaction_consumed=False)
        assert plan_workflow(stored, shards, resolve_preset=_resolve).steps[0].action == action
