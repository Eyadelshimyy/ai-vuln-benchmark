"""VibeBB Record Protocol v2: typed writers, stdlib hook mirror and Stop enforcement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from fpga import _vrp, records

SCRIPTS = Path(__file__).parents[1] / "plugins" / "fpga" / "hooks" / "scripts"
HOOK = SCRIPTS / "require_records.py"
BLIND_GUARD = SCRIPTS / "blind_guard.py"

IMPRESSION = (
    "The harness drawing reads as a coherent build sheet: wire W1 runs from C1.1 to "
    "C2.1 and every cavity carries a unique label. What worries me is the 3.3 V sense "
    "lead, which runs parallel to the motor feed for most of its length and could pick "
    "up switching noise. A fabricator would still be able to cut and crimp from the "
    "tables alone, although the splice note is small. Next I would re-route the sense "
    "lead along the far edge of the bundle and re-render to confirm the separation. "
    "Overall the sheet communicates intent well, and the remaining risk sits in one "
    "routing choice."
)
IMPRESSION_B = (
    "After the regeneration, W1 still lands on C2.1 but the bundle now leaves the "
    "enclosure through the lower grommet, which shortens the exposed run by a hand "
    "width. The crimp table lists every terminal and the cut lengths add up against "
    "the route plan, so the kitting step should be quick. I am less sure about the "
    "strain relief at the grommet because the contract gives no bend radius for the "
    "jacket. Somebody assembling it would appreciate the clearer branch order, while "
    "the end user never sees any of it unless the lid is opened for service later."
)
IMPRESSION_C = (
    "Third pass on the same harness: the BOM rows for C1.1 and C2.1 now carry "
    "manufacturer part numbers instead of family names, which removes a purchasing "
    "question entirely. Splice S1 moved off the flex zone near the hinge, so fatigue "
    "is no longer the dominant risk I see here. The drawing gained a test-point table "
    "that production can use for continuity checks without a second document. My "
    "remaining doubt concerns colour coding under monochrome printing on the shop "
    "floor, which I will verify with a grayscale export before handing it over."
)
IMPRESSION_JA = (
    "電源ペアと信号ペアが明確に分かれており、配線表だけで切断・圧着まで進められる構成だと感じた。"
    "一方でW1のセンス線がモータ給電線と長く並走しており、スイッチングノイズを拾う懸念が残る。"
    "製造現場の作業者は端子番号を迷わず読めるが、スプライス注記の文字が小さく見落とされやすい。"
    "次の工程ではセンス線を束の反対側へ移し、再描画して分離が保たれているかを確認したい。"
    "全体としては設計意図が伝わる図面であり、誤読のリスクは注記の大きさに集中している。"
    "コネクタの嵌合方向は矢印で示されているが、裏面視か表面視かの明記が無いので追記したい。"
    "ケーブル長の基準点は嵌合面に揃っており、治具板への展開もそのまま行えると判断した。"
    "保守担当者が現地で断線を探す場面を想像すると、試験点の一覧が図面に同居している点は心強い。"
    "ただしモノクロ印刷では電線色の区別が失われるため、色名の略号を併記するかどうかを次に検討したい。"
    "圧着端子の品番は部品表と一致しており、購買部門が型番を問い合わせる手間も生じないと考える。"
)
FELT = (
    "The verse about the long parallel run made the sense lead feel fragile rather than "
    "merely suboptimal. It is a song, not a gate, but it is a reason to look again."
)


def _hook_module() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("_records", SCRIPTS / "_records.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOOK_RECORDS = _hook_module()


def _facets(**overrides: Any) -> dict[str, Any]:
    facets: dict[str, Any] = {
        "observed": ["W1 runs C1.1 to C2.1", "sense lead parallels the motor feed"],
        "works": ["cavity labels are unique"],
        "concerns": [
            {
                "id": "c1",
                "text": "sense lead couples motor switching noise",
                "severity": "warning",
                "about": "self",
                "anchor": "W1",
            }
        ],
        "feelings": {
            "maker": "can kit and crimp from the tables",
            "user": "never sees it unless servicing",
        },
        "next_actions": ["re-route the sense lead along the far edge"],
    }
    return facets | overrides


def _impression(
    text: str = IMPRESSION, artifacts: list[str] | None = None, **overrides: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": "export",
        "artifacts": artifacts or ["out"],
        "impression": text,
        "facets": _facets(),
        "claims": [
            {
                "text": "wire W1 exists in the wire list",
                "anchor": "W1",
                "artifact": "out/fpga-reports/demo.fpga-report.json",
            },
            {
                "text": "C1.1 is the source cavity",
                "anchor": "C1.1",
                "artifact": "out/fpga-reports/demo.fpga-report.json",
            },
        ],
        "confidence": "medium",
        "unknowns": ["actual motor current ripple"],
    }
    return payload | overrides


def _vision(reviewer: str = "primary", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_path": "out/fpga-reports/demo.fpga-pinmap.png",
        "model": "openhands/kimi-k3",
        "checklist": "harness-diagram",
        "reviewer": reviewer,
        "findings": [{"category": "label_collision", "severity": "warning", "note": "W1"}],
        "impression": IMPRESSION,
        "facets": _facets(),
        "claims": [
            {
                "text": "W1 label is drawn",
                "anchor": "W1",
                "artifact": "out/fpga-reports/demo.fpga-pinmap.png",
            }
        ],
        "lookback": [{"claim": 0, "verdict": "confirmed", "note": "second look shows W1"}],
        "confidence": "medium",
        "unknowns": [],
    }
    return payload | overrides


LIST_CLAIM = [
    {
        "text": "W1 is in the wire list",
        "anchor": "W1",
        "artifact": "out/fpga-reports/demo.fpga-report.json",
    }
]


def _decision(evidence_path: str) -> dict[str, Any]:
    return {
        "id": "sense-lead-routing",
        "stage": "design",
        "question": "Where should the 3.3 V sense lead run inside the bundle?",
        "principles": [
            "Inductive coupling falls with separation and shorter parallel run length",
            "IPC/WHMA-A-620 segregation of power and signal conductors",
        ],
        "options": [
            {"name": "same-bundle", "pros": ["shortest"], "cons": ["noise coupling"]},
            {"name": "far-edge", "pros": ["separation"], "cons": ["+40 mm length"]},
        ],
        "chosen": "far-edge",
        "rationale": (
            "The sense lead feeds an ADC whose LSB is about 0.8 mV, while the motor feed "
            "switches several amperes at 20 kHz; mutual inductance along a 300 mm parallel "
            "run would inject more than an LSB, so physical separation is the cheapest fix."
        ),
        "evidence": [{"path": evidence_path}, {"reference": "IPC/WHMA-A-620E section 13"}],
        "assumptions": ["motor PWM is 20 kHz"],
        "unknowns": ["actual motor current ripple"],
        "risks": ["longer lead adds 40 mm of copper"],
        "revisit_when": "a measured ADC noise floor exceeds 2 LSB",
    }


def _sister(root: Path, system: str, kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """Append a valid v2 record to another plugin's log, as that plugin's writer would."""
    log = root / "observations" / system / _vrp.LOG_FILES[kind]
    log.parent.mkdir(parents=True, exist_ok=True)
    existing, _ = _vrp.load_jsonl(log)
    record: dict[str, Any] = {
        "schema_version": 2,
        "kind": kind,
        "plugin": system,
        "sequence": len(existing) + 1,
        "prev_event_id": _vrp.chain_head(existing),
        "recorded_at": datetime.now(UTC).isoformat(),
        "writer": {"plugin": system},
        **body,
    }
    record["event_id"] = _vrp.compute_event_id(record)
    assert _vrp.record_errors(kind, record) == []
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _sister_impression(
    root: Path, system: str, artifact: str, text: str, **body: Any
) -> dict[str, Any]:
    path = root / artifact
    claims = [
        {"text": "the export names the connector", "anchor": "J1", "artifact": artifact},
        {"text": "the export names the net", "anchor": "VBUS", "artifact": artifact},
    ]
    payload: dict[str, Any] = {
        "stage": "export",
        "artifacts": [{"path": artifact, "sha256": _vrp.tree_sha256(path)}],
        "impression": text,
        "facets": _facets(
            concerns=[
                {
                    "id": "c1",
                    "text": "J1 pin 3 carries VBUS next to a signal pin",
                    "severity": "error",
                    "about": "fpga",
                    "anchor": "J1",
                }
            ]
        ),
        "claims": claims,
        "confidence": "high",
        "unknowns": [],
    }
    return _sister(root, system, "stage_impression", payload | body)


SISTER_TEXT = (
    "The fpga export names J1 as the harness connector and routes VBUS to pin 3, "
    "directly beside the I2C clock on pin 4. That arrangement works electrically, yet a "
    "mis-mated plug would put five volts onto a logic input with nothing to stop it. "
    "The wire team should key the housing or move VBUS to the outer cavity. A maker will "
    "find the netlist easy to follow, and a user will only notice if a repair goes wrong."
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    (tmp_path / "out" / "fpga-reports").mkdir(parents=True)
    (tmp_path / "out" / "fpga-reports" / "demo.fpga-report.json").write_text(
        "W1,C1.1,C2.1\n", encoding="utf-8"
    )
    (tmp_path / "out" / "fpga-reports" / "demo.fpga-pinmap.png").write_bytes(b"\x89PNG fake")
    return tmp_path


def _connectivity(root: Path) -> Path:
    path = root / "fpga" / "board.connectivity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"connectors": ["J1"], "nets": ["VBUS"]}\n', encoding="utf-8")
    return path


# --- text and structure -------------------------------------------------


def test_impression_rules() -> None:
    assert records.impression_is_prose(IMPRESSION) == IMPRESSION
    assert records.impression_is_prose(IMPRESSION_JA) == IMPRESSION_JA
    with pytest.raises(ValueError, match="characters"):
        records.impression_is_prose("Looks fine. No issues. Done.")
    with pytest.raises(ValueError, match="sentences"):
        records.impression_is_prose("x" * 500 + ". y")
    with pytest.raises(ValueError, match="repeats"):
        records.impression_is_prose("The drawing is fine and readable overall. " * 12)
    assert records.sentence_count("Supply is 3.3 V and 5.0 V") == 0
    assert HOOK_RECORDS.impression_errors(IMPRESSION) == []
    assert HOOK_RECORDS.impression_errors("Looks fine. No issues. Done.")


def test_vendored_core_matches_hook() -> None:
    hook = (SCRIPTS / "_records.py").read_bytes()
    assert (Path(records.__file__).parent / "_vrp.py").read_bytes() == hook


def test_record_impression_chains_and_binds(workspace: Path) -> None:
    first = records.record_impression(_impression())["record"]
    assert first["artifacts"] == [{"path": "out", "sha256": records.tree_sha256(workspace / "out")}]
    assert first["sequence"] == 1
    assert first["prev_event_id"] == _vrp.GENESIS
    assert first["event_id"] == _vrp.compute_event_id(first)
    assert first["writer"] == {"plugin": "fpga"}
    assert HOOK_RECORDS.record_errors("stage_impression", first) == []
    second = records.record_impression(_impression(IMPRESSION_B))["record"]
    assert second["sequence"] == 2
    assert second["prev_event_id"] == first["event_id"]
    with pytest.raises(ValueError, match="outside the workspace"):
        records.record_impression(_impression(IMPRESSION_C, artifacts=["/etc/passwd"]))
    with pytest.raises(ValueError, match="does not exist"):
        records.record_impression(_impression(IMPRESSION_C, artifacts=["out/missing.csv"]))


def test_writer_metadata_from_env(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRP_AGENT", "fpga-developer")
    monkeypatch.setenv("VRP_MODEL", "openhands/kimi-k3")
    record = records.record_impression(_impression())["record"]
    assert record["writer"] == {
        "plugin": "fpga",
        "agent": "fpga-developer",
        "model": "openhands/kimi-k3",
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"facets": _facets(observed=["only one thing seen"])}, "observed"),
        ({"facets": _facets(works=[])}, "works"),
        ({"facets": _facets(concerns=[])}, "no_concerns_reason"),
        ({"facets": _facets(next_actions=[])}, "next_actions"),
        ({"claims": []}, "claims"),
        ({"confidence": "certain"}, "confidence"),
    ],
)
def test_structured_facets_are_required(
    workspace: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValidationError, ValueError), match=message):
        records.record_impression(_impression(**change))
    record = _impression(**change) | {"kind": "stage_impression"}
    assert any(message in e for e in HOOK_RECORDS.impression_body_errors(record, min_claims=2))


def test_no_concerns_needs_a_reason(workspace: Path) -> None:
    reason = "every gate passes with margin and the drawing was checked twice"
    facets = _facets(concerns=[], no_concerns_reason=reason)
    assert records.record_impression(_impression(facets=facets))["verdict"] == "pass"


def test_claims_must_be_grounded(workspace: Path) -> None:
    ghost = [
        {
            "text": "wire W9 exists in the list",
            "anchor": "W9",
            "artifact": "out/fpga-reports/demo.fpga-report.json",
        },
        {
            "text": "C1.1 is the source cavity",
            "anchor": "C1.1",
            "artifact": "out/fpga-reports/demo.fpga-report.json",
        },
    ]
    with pytest.raises(ValueError, match="'W9' does not occur"):
        records.record_impression(_impression(claims=ghost))
    elsewhere = [
        {
            "text": "wire W1 is in the list",
            "anchor": "W1",
            "artifact": "out/fpga-reports/demo.fpga-report.json",
        },
        {"text": "something outside", "anchor": "root", "artifact": "README.md"},
    ]
    with pytest.raises(ValueError, match="not among the bound artifacts"):
        records.record_impression(_impression(claims=elsewhere))
    unanchored = IMPRESSION.replace("W1", "the wire").replace("C1.1", "one cavity")
    with pytest.raises(ValueError, match="mention at least one claim anchor"):
        records.record_impression(_impression(unanchored))


def test_near_duplicate_is_rejected_and_delta_required(workspace: Path) -> None:
    records.record_impression(_impression())
    with pytest.raises(ValueError, match="similar to an earlier one; read"):
        records.record_impression(_impression(IMPRESSION + " Nothing else."))
    parts = _vrp.sentences(IMPRESSION)
    parts[3] = _vrp.sentences(IMPRESSION_B)[0]
    half = " ".join(parts)
    with pytest.raises(ValueError, match="add delta"):
        records.record_impression(_impression(half))
    delta = "the grommet exit moved and the exposed run is now one hand width shorter"
    assert records.record_impression(_impression(half, delta=delta))["verdict"] == "pass"


# --- chain integrity -----------------------------------------------------


def _log(workspace: Path, kind: str = "stage_impression") -> Path:
    return workspace / records.RECORDS_DIR / records.LOG_FILES[kind]


def _rewrite(path: Path, lines: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


def _three(workspace: Path) -> list[dict[str, Any]]:
    for text in (IMPRESSION, IMPRESSION_B, IMPRESSION_C):
        records.record_impression(_impression(text))
    rows, _ = _vrp.load_jsonl(_log(workspace))
    return rows


def _rehash(record: dict[str, Any]) -> dict[str, Any]:
    return record | {"event_id": _vrp.compute_event_id(record)}


def test_chain_detects_edit(workspace: Path) -> None:
    rows = _three(workspace)
    rows[1]["impression"] = rows[1]["impression"].replace("hand", "foot")
    _rewrite(_log(workspace), rows)
    problems = _vrp.verify_log(_log(workspace), "stage_impression")[1]
    assert any("line 2: event_id does not match" in p for p in problems)


def test_chain_detects_relink(workspace: Path) -> None:
    rows = _three(workspace)
    rows[2] = _rehash(rows[2] | {"prev_event_id": "f" * 64})
    _rewrite(_log(workspace), rows)
    problems = _vrp.verify_log(_log(workspace), "stage_impression")[1]
    assert any("line 3: prev_event_id" in p for p in problems)


def test_chain_detects_deletion(workspace: Path) -> None:
    rows = _three(workspace)
    _rewrite(_log(workspace), [rows[0], rows[2]])
    problems = _vrp.verify_log(_log(workspace), "stage_impression")[1]
    assert any("sequence 3" in p for p in problems)
    assert any("prev_event_id" in p for p in problems)


def test_chain_detects_time_regression(workspace: Path) -> None:
    rows = _three(workspace)
    earlier = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    rows[2] = _rehash(rows[2] | {"recorded_at": earlier})
    _rewrite(_log(workspace), rows)
    problems = _vrp.verify_log(_log(workspace), "stage_impression")[1]
    assert any("goes back in time" in p for p in problems)


def test_chain_detects_duplicate_and_malformed(workspace: Path) -> None:
    rows = _three(workspace)
    _rewrite(_log(workspace), [rows[0], rows[0]])
    with _log(workspace).open("a", encoding="utf-8") as stream:
        stream.write("{not json\n")
    problems = _vrp.verify_log(_log(workspace), "stage_impression")[1]
    assert any("duplicate event_id" in p for p in problems)
    assert any("malformed" in p for p in problems)


def test_writer_refuses_to_extend_a_broken_chain(workspace: Path) -> None:
    rows = _three(workspace)
    rows[0]["stage"] = "brief"
    _rewrite(_log(workspace), rows)
    with pytest.raises(ValueError, match="fail integrity"):
        records.record_impression(_impression(IMPRESSION_JA))
    assert (
        records.record_decision(_decision("out/fpga-reports/demo.fpga-report.json"))["verdict"]
        == "pass"
    )
    assert records.records_summary()["verdict"] == "unknown"


def test_legacy_v1_log_is_archived(workspace: Path) -> None:
    log = _log(workspace)
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"schema_version": 1, "kind": "stage_impression"}) + "\n")
    assert _vrp.verify_log(log, "stage_impression") == ([], [])
    records.record_impression(_impression())
    assert (log.parent / "impressions.v1.jsonl").is_file()
    assert _vrp.verify_log(log, "stage_impression")[1] == []


# --- decisions ------------------------------------------------------------


def test_record_decision_requires_principled_choice(workspace: Path) -> None:
    record = records.record_decision(_decision("out/fpga-reports/demo.fpga-report.json"))["record"]
    digest = hashlib.sha256(
        (workspace / "out" / "fpga-reports" / "demo.fpga-report.json").read_bytes()
    ).hexdigest()
    assert record["evidence"][0] == {
        "path": "out/fpga-reports/demo.fpga-report.json",
        "sha256": digest,
    }
    assert record["evidence"][1]["reference"].startswith("IPC")
    assert HOOK_RECORDS.record_errors("decision", record) == []
    for key, value in (
        ("options", _decision("out/fpga-reports/demo.fpga-report.json")["options"][:1]),
        ("chosen", "teleport"),
        ("rationale", "because"),
        ("principles", []),
        ("principles", ["vibes"]),
        ("risks", []),
        ("evidence", []),
    ):
        with pytest.raises(ValidationError):
            records.record_decision(
                _decision("out/fpga-reports/demo.fpga-report.json") | {key: value}
            )
    with pytest.raises(ValidationError):
        records.record_decision(
            _decision("out/fpga-reports/demo.fpga-report.json") | {"surprise": 1}
        )


def test_decision_impression_refs_must_resolve(workspace: Path) -> None:
    _connectivity(workspace)
    sister = _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    good = {"system": "fpga", "event_id": sister["event_id"]}
    record = records.record_decision(
        _decision("out/fpga-reports/demo.fpga-report.json") | {"impression_refs": [good]}
    )
    assert record["record"]["impression_refs"] == [good]
    bad = {"system": "fpga", "event_id": "a" * 64}
    with pytest.raises(ValueError, match="unresolved refs"):
        records.record_decision(
            _decision("out/fpga-reports/demo.fpga-report.json") | {"impression_refs": [bad]}
        )


# --- reading sister impressions --------------------------------------------


def test_upstream_must_resolve(workspace: Path) -> None:
    _connectivity(workspace)
    sister = _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    effect = "keyed the J1 housing so a mis-mate cannot put VBUS on a signal cavity"
    entry = {
        "system": "fpga",
        "event_id": sister["event_id"],
        "disposition": "adopted",
        "effect": effect,
        "concern_ids": ["c1"],
    }
    assert records.record_impression(_impression(upstream=[entry]))["verdict"] == "pass"
    for broken, message in (
        (entry | {"event_id": "b" * 64}, "has no impression"),
        (entry | {"system": "sim"}, "no records for sim"),
        (entry | {"concern_ids": ["c9"]}, "are not in that impression"),
    ):
        with pytest.raises(ValueError, match=message):
            records.record_impression(_impression(IMPRESSION_B, upstream=[broken]))
    log = workspace / "observations" / "fpga" / "impressions.jsonl"
    log.write_text(log.read_text(encoding="utf-8").replace("VBUS", "VDD", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="fail integrity"):
        records.record_impression(_impression(IMPRESSION_C, upstream=[entry]))


def test_read_receipt_resolves_bound_impressions(workspace: Path) -> None:
    source = _connectivity(workspace)
    sister = _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    read = records.record_read({"artifact": str(source), "producer": "fpga"})["record"]
    assert read["refs"] == [{"system": "fpga", "event_id": sister["event_id"]}]
    assert read["unresolved"] == []
    mech = workspace / "mech.envelope.json"
    mech.write_text("{}", encoding="utf-8")
    lonely = records.record_read({"artifact": "mech.envelope.json", "producer": "mech"})
    assert lonely["record"]["refs"] == []
    assert lonely["record"]["unresolved"]


def test_explicit_impression_refs_in_artifact(workspace: Path) -> None:
    source = _connectivity(workspace)
    sister = _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    refs_in_doc: list[dict[str, str]] = [{"system": "fpga", "event_id": sister["event_id"]}]
    doc = {"connectors": ["J1"], "impression_refs": refs_in_doc}
    source.write_text(json.dumps(doc), encoding="utf-8")
    refs, unresolved = records.impressions_for_artifact(source, "fpga")
    assert refs == [{"system": "fpga", "event_id": sister["event_id"]}]
    assert unresolved == []
    refs_in_doc.append({"system": "fpga", "event_id": "c" * 64})
    source.write_text(json.dumps(doc), encoding="utf-8")
    _, unresolved = records.impressions_for_artifact(source, "fpga")
    assert any("has no impression" in u for u in unresolved)


# --- loop guards ------------------------------------------------------------


def test_disputed_concern_cannot_ping_pong(workspace: Path) -> None:
    mine = records.record_impression(
        _impression(
            facets=_facets(
                concerns=[
                    {
                        "id": "c1",
                        "text": "fpga routes VBUS beside the I2C clock on J1",
                        "severity": "error",
                        "about": "fpga",
                        "anchor": "W1",
                    }
                ]
            )
        )
    )["record"]
    effect = "VBUS beside SCL is protected by the keyed housing, so no layout change"
    _sister_impression(
        workspace,
        "fpga",
        "out/fpga-reports/demo.fpga-report.json",
        SISTER_TEXT.replace("J1", "W1"),
        claims=[
            {
                "text": "W1 is listed",
                "anchor": "W1",
                "artifact": "out/fpga-reports/demo.fpga-report.json",
            },
            {
                "text": "C2.1 is listed",
                "anchor": "C2.1",
                "artifact": "out/fpga-reports/demo.fpga-report.json",
            },
        ],
        upstream=[
            {
                "system": "fpga",
                "event_id": mine["event_id"],
                "disposition": "disputed",
                "effect": effect,
                "concern_ids": ["c1"],
            }
        ],
    )
    again = _facets(
        concerns=[
            {
                "id": "c2",
                "text": "fpga routes VBUS beside the I2C clock on J1 again",
                "severity": "error",
                "about": "fpga",
            }
        ]
    )
    with pytest.raises(ValueError, match="repeats one fpga disputed"):
        records.record_impression(_impression(IMPRESSION_B, facets=again))
    (workspace / "out" / "fpga-reports" / "demo.fpga-report.json").write_text(
        "W1,C1.1,C2.1\nW2,C1.2,C2.2\n"
    )
    assert records.record_impression(_impression(IMPRESSION_B, facets=again))["verdict"] == "pass"


def _with_insight(text: str, **insight: Any) -> dict[str, Any]:
    base = {
        "id": "i1",
        "hypothesis": "a keyed J1 housing removes the mis-mate risk without moving VBUS",
        "proposed_change": "switch J1 to the keyed variant",
        "expected_effect": "mis-mate becomes mechanically impossible",
        "test": "connector keying gate on the contract",
        "target": "fpga",
    }
    return _impression(text, insights=[base | insight])


def test_insight_lifecycle_and_no_rejected_reproposal(workspace: Path) -> None:
    origin = records.record_impression(_with_insight(IMPRESSION))["record"]
    ref = {"system": "fpga", "event_id": origin["event_id"], "insight_id": "i1"}
    reason = "the keyed variant is not stocked in the required 2.54 mm pitch at all"
    evidence = [{"reference": "vendor catalogue 2026"}]
    with pytest.raises(ValueError, match="cannot move from proposed to adopted"):
        records.record_insight(
            {
                "insight": ref,
                "status": "adopted",
                "reason": reason,
                "evidence": evidence,
                "gate_verdicts": [{"gate": "keying", "verdict": "pass"}],
                "decision_refs": ["d" * 64],
            }
        )
    records.record_insight(
        {"insight": ref, "status": "tried", "reason": reason, "evidence": evidence}
    )
    with pytest.raises(ValueError, match="deterministic gate verdict of pass"):
        records.record_insight(
            {"insight": ref, "status": "adopted", "reason": reason, "evidence": evidence}
        )
    records.record_insight(
        {"insight": ref, "status": "rejected", "reason": reason, "evidence": evidence}
    )
    with pytest.raises(ValueError, match="cannot move from rejected"):
        records.record_insight(
            {"insight": ref, "status": "tried", "reason": reason, "evidence": evidence}
        )
    with pytest.raises(ValueError, match="repeats rejected"):
        records.record_impression(_with_insight(IMPRESSION_B, id="i2"))
    revisit = _with_insight(
        IMPRESSION_B,
        id="i2",
        revisits=ref,
        revisit_reason="a second source now stocks the keyed housing in 2.54 mm pitch",
    )
    assert records.record_impression(revisit)["verdict"] == "pass"
    digest = _vrp.digest(workspace)
    assert {i["status"] for i in digest["insights"]} == {"rejected", "proposed"}


def test_blind_review_rounds_are_finite(workspace: Path) -> None:
    with pytest.raises(ValueError, match="needs a primary review"):
        records.record_vision_review(_vision("blind", impression=IMPRESSION_B))
    primary = records.record_vision_review(_vision())["record"]
    blind = records.record_vision_review(
        _vision(
            "blind",
            impression=IMPRESSION_B.replace("W1", "W1 "),
            findings=[{"category": "missing_dimension", "severity": "error", "note": "S1"}],
        )
    )["record"]
    with pytest.raises(ValueError, match="already has its blind review"):
        records.record_vision_review(_vision("blind", impression=IMPRESSION_C + " W1."))
    with pytest.raises(ValueError, match="only allowed after a round-1 reconcile"):
        records.record_vision_review(_vision("tiebreak", impression=IMPRESSION_C + " W1."))
    first = records.record_reconcile({"reviews": [primary["event_id"], blind["event_id"]]})
    assert first["record"]["outcome"] == "disagree"
    with pytest.raises(ValueError, match="already reconciled"):
        records.record_reconcile({"reviews": [primary["event_id"], blind["event_id"]]})
    tiebreak = records.record_vision_review(
        _vision(
            "tiebreak",
            impression=IMPRESSION_C + " W1 stays.",
            findings=[{"category": "design_intent", "severity": "info", "note": "x"}],
        )
    )["record"]
    with pytest.raises(ValueError, match="already has its tiebreak"):
        records.record_vision_review(_vision("tiebreak", impression=IMPRESSION_JA))
    ids = [primary["event_id"], blind["event_id"], tiebreak["event_id"]]
    final = records.record_reconcile({"reviews": ids})["record"]
    assert final["round"] == 2
    assert final["outcome"] == "unresolved"
    with pytest.raises(ValueError, match="already reconciled"):
        records.record_reconcile({"reviews": ids})


def test_vision_lookback_covers_every_claim(workspace: Path) -> None:
    with pytest.raises(ValueError, match="lookback misses claims"):
        records.record_vision_review(_vision(lookback=[]))
    refuted = [{"claim": 0, "verdict": "refuted", "note": "W1 label is actually W7"}]
    with pytest.raises(ValueError, match="refuted claim must surface"):
        records.record_vision_review(_vision(lookback=refuted, findings=[]))
    event = records.record_vision_review(
        _vision(image_path=None, source_event_id="a" * 64, claims=LIST_CLAIM)
    )
    assert event["record"].get("image_sha256") is None


def _song(root: Path, to: str = "fpga", song_id: str = "ode-to-w1") -> Path:
    verse = "Along the bundle W1 hums, beside the motor's restless drums, keep it apart."
    doc = {
        "kind": "bard_song_delivery",
        "schema_version": 1,
        "id": song_id,
        "system": "bard",
        "to": to,
        "title": "Ode to W1",
        "verse": verse,
        "song": {
            "path": "out/fpga-reports/demo.fpga-report.json",
            "sha256": _vrp.sha256_file(root / "out" / "fpga-reports" / "demo.fpga-report.json"),
        },
        "impression_refs": [{"system": "fpga", "event_id": "e" * 64}],
        "reason": "wire carries the sense lead the song is about",
        "created_at": datetime.now(UTC).isoformat(),
    }
    path = root / "liaison" / f"{song_id}.bard-song.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_song_receipt_never_calls_bard_back(workspace: Path) -> None:
    _song(workspace)
    receipt = {
        "delivery": "liaison/ode-to-w1.bard-song.json",
        "felt": FELT,
        "prompted_review": True,
        "follow_up": "re-check the W1 parallel run length",
    }
    record = records.record_song_receipt(receipt)["record"]
    assert record["song_id"] == "ode-to-w1"
    with pytest.raises(ValueError, match="already has a receipt"):
        records.record_song_receipt(receipt)
    _song(workspace, to="mech", song_id="ode-to-lid")
    with pytest.raises(ValueError, match="addressed to mech"):
        records.record_song_receipt(receipt | {"delivery": "liaison/ode-to-lid.bard-song.json"})
    assert not (workspace / "observations" / "bard").exists()
    assert sorted(p.name for p in (workspace / "liaison").iterdir()) == [
        "ode-to-lid.bard-song.json",
        "ode-to-w1.bard-song.json",
    ]
    with pytest.raises(ValidationError):
        records.record_song_receipt(receipt | {"compose_reply": True})


# --- cross-plugin search ------------------------------------------------------


def test_digest_and_search_across_plugins(workspace: Path) -> None:
    _connectivity(workspace)
    sister = _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    records.record_impression(_impression())
    digest = records.records_digest()
    refs = {c["ref"] for c in digest["open_concerns"]}
    assert f"fpga:{sister['event_id']}#c1" in refs
    assert digest["open_concerns"][0]["severity"] == "error"
    assert digest["systems"]["fpga"]["integrity"] == "ok"
    hits = records.records_search({"query": "VBUS", "severity": "error"})["results"]
    assert [h["system"] for h in hits] == ["fpga"]
    by_artifact = records.records_search({"artifact": "out"})["results"]
    assert [h["system"] for h in by_artifact] == ["fpga"]
    with pytest.raises(ValueError, match="unknown search fields"):
        records.records_search({"sql": "drop"})
    log = workspace / "observations" / "fpga" / "impressions.jsonl"
    log.write_text(log.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8")
    result = records.records_search({"query": "VBUS"})
    assert result["verdict"] == "unknown"
    assert result["results"] == []
    assert "fpga" in result["integrity_problems"]


# --- hook mirror ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("decision", "principles"),
        ("decision", "options"),
        ("decision", "rationale"),
        ("decision", "evidence"),
        ("decision", "risks"),
        ("decision", "revisit_when"),
        ("decision", "unknowns"),
        ("decision", "prev_event_id"),
        ("stage_impression", "impression"),
        ("stage_impression", "artifacts"),
        ("stage_impression", "facets"),
        ("stage_impression", "claims"),
        ("stage_impression", "writer"),
        ("vision_review", "impression"),
        ("vision_review", "model"),
        ("vision_review", "lookback"),
        ("vision_review", "recorded_at"),
        ("vision_review", "event_id"),
        ("song_receipt", "felt"),
        ("song_receipt", "delivery"),
    ],
)
def test_hook_mirror_rejects_mutations(workspace: Path, kind: str, field: str) -> None:
    def song() -> dict[str, Any]:
        _song(workspace)
        return records.record_song_receipt(
            {"delivery": "liaison/ode-to-w1.bard-song.json", "felt": FELT, "prompted_review": False}
        )

    writers = {
        "decision": lambda: records.record_decision(
            _decision("out/fpga-reports/demo.fpga-report.json")
        ),
        "stage_impression": lambda: records.record_impression(_impression()),
        "vision_review": lambda: records.record_vision_review(_vision()),
        "song_receipt": song,
    }
    record = dict(writers[kind]()["record"])
    assert HOOK_RECORDS.record_errors(kind, record) == []
    del record[field]
    assert HOOK_RECORDS.record_errors(kind, record)
    record[field] = ""
    assert HOOK_RECORDS.record_errors(kind, record)


# --- Stop hook ---------------------------------------------------------------


def _hook(mode: str, root: Path, session: str = "s1") -> subprocess.CompletedProcess[str]:
    env = dict(os.environ) | {"OPENHANDS_PROJECT_DIR": str(root)}
    return subprocess.run(
        [sys.executable, str(HOOK), mode],
        input=json.dumps({"session_id": session, "working_dir": str(root)}),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _status(root: Path) -> dict[str, Any]:
    path = root / records.RECORDS_DIR / "records-status.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_stop_without_marker_allows(tmp_path: Path) -> None:
    assert _hook("stop", tmp_path).returncode == 0


def test_stop_clean_session_passes(tmp_path: Path) -> None:
    assert _hook("session-start", tmp_path).returncode == 0
    result = _hook("stop", tmp_path)
    assert result.returncode == 0, result.stdout
    assert _status(tmp_path)["verdict"] == "pass"


def test_stop_enforces_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    assert _hook("session-start", tmp_path).returncode == 0
    out = tmp_path / "out"
    (out / "fpga-reports").mkdir(parents=True)
    (out / "fpga-reports" / "demo.fpga-report.json").write_text("W1,C1.1,C2.1\n", encoding="utf-8")
    log = tmp_path / records.RECORDS_DIR
    event = {"event_id": "b" * 64, "question": "is it legible?", "session_id": "s1"}
    (log / "vision-tool-events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    denied = _hook("stop", tmp_path)
    assert denied.returncode == 2
    payload = json.loads(denied.stdout)
    assert payload["decision"] == "deny"
    assert "out/fpga-reports/demo.fpga-report.json" in payload["additionalContext"]
    assert "vision tool event" in payload["additionalContext"]
    assert "no decision record" in payload["additionalContext"]
    assert "refusal 1/2" in payload["additionalContext"]
    assert _status(tmp_path)["verdict"] == "fail"

    records.record_decision(_decision("out/fpga-reports/demo.fpga-report.json"))
    records.record_impression(_impression())
    records.record_vision_review(
        _vision(image_path=None, source_event_id="b" * 64, claims=LIST_CLAIM)
    )
    passed = _hook("stop", tmp_path)
    assert passed.returncode == 0, passed.stdout
    assert _status(tmp_path)["verdict"] == "pass"

    (out / "fpga-reports" / "demo.fpga-report.json").write_text(
        "W1,C1.1,C2.1\nW2,C1.2,C2.2\n", encoding="utf-8"
    )
    stale = _hook("stop", tmp_path)
    assert stale.returncode == 2
    assert "no fresh stage_impression" in json.loads(stale.stdout)["reason"]

    released = _hook("stop", tmp_path)
    assert released.returncode == 0
    assert json.loads(released.stdout)["decision"] == "allow"
    assert "still unmet" in json.loads(released.stdout)["additionalContext"]
    third = _hook("stop", tmp_path)
    assert third.returncode == 0


def test_stop_flags_broken_chain(tmp_path: Path) -> None:
    assert _hook("session-start", tmp_path).returncode == 0
    log = tmp_path / records.RECORDS_DIR
    forged = {
        "schema_version": 2,
        "kind": "stage_impression",
        "plugin": "fpga",
        "sequence": 1,
        "prev_event_id": _vrp.GENESIS,
        "event_id": "c" * 64,
        "recorded_at": "2999-01-01T00:00:00+00:00",
        "writer": {"plugin": "fpga"},
        "stage": "export",
        "artifacts": [{"path": "out", "sha256": "d" * 64}],
        "impression": "ok.",
    }
    (log / "impressions.jsonl").write_text(json.dumps(forged) + "\n{not json\n", encoding="utf-8")
    result = _hook("stop", tmp_path)
    assert result.returncode == 2
    reason = json.loads(result.stdout)["reason"]
    assert "malformed" in reason
    assert "event_id does not match" in reason
    assert "invalid stage_impression" in reason


def test_viewed_image_needs_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    assert _hook("session-start", tmp_path).returncode == 0
    log = tmp_path / records.RECORDS_DIR
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8jpg")
    digest = hashlib.sha256(b"\xff\xd8jpg").hexdigest()
    observation = {
        "event_id": "e" * 64,
        "image_path": str(image),
        "image_sha256": digest,
        "session_id": "s1",
    }
    (log / "image-observations.jsonl").write_text(json.dumps(observation) + "\n", encoding="utf-8")
    assert _hook("stop", tmp_path).returncode == 2
    claim = [{"text": "W1 is drawn on it", "anchor": "W1", "artifact": "photo.jpg"}]
    records.record_vision_review(
        _vision(image_path="photo.jpg", checklist="intake-image", claims=claim)
    )
    assert _hook("stop", tmp_path).returncode == 0


def test_stop_requires_upstream_answer(workspace: Path) -> None:
    assert _hook("session-start", workspace).returncode == 0
    source = _connectivity(workspace)
    sister = _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    records.record_read({"artifact": str(source), "producer": "fpga"})
    denied = _hook("stop", workspace)
    assert denied.returncode == 2
    assert f"sister impression fpga:{sister['event_id'][:12]}" in denied.stdout
    effect = "keyed the J1 housing so a mis-mate cannot put VBUS on a signal cavity"
    upstream = [
        {
            "system": "fpga",
            "event_id": sister["event_id"],
            "disposition": "noted",
            "effect": effect,
        }
    ]
    records.record_decision(_decision("out/fpga-reports/demo.fpga-report.json"))
    records.record_impression(_impression(upstream=upstream))
    assert _hook("stop", workspace).returncode == 0


def test_stop_requires_dual_review_then_terminates(workspace: Path) -> None:
    assert _hook("session-start", workspace).returncode == 0
    records.record_decision(_decision("out/fpga-reports/demo.fpga-report.json"))
    records.record_impression(_impression())
    primary = records.record_vision_review(_vision())["record"]
    first = _hook("stop", workspace)
    assert first.returncode == 2
    assert "needs a blind second review" in first.stdout
    blind = records.record_vision_review(
        _vision(
            "blind",
            impression=IMPRESSION_B,
            findings=[{"category": "missing_dimension", "severity": "error", "note": "S1"}],
            claims=[
                {
                    "text": "W1 is drawn",
                    "anchor": "W1",
                    "artifact": "out/fpga-reports/demo.fpga-pinmap.png",
                }
            ],
        )
    )["record"]
    records.record_reconcile({"reviews": [primary["event_id"], blind["event_id"]]})
    second = _hook("stop", workspace)
    assert second.returncode == 2
    assert "one tiebreak review" in second.stdout
    tiebreak = records.record_vision_review(
        _vision(
            "tiebreak",
            impression=IMPRESSION_C + " W1 stays.",
            findings=[{"category": "design_intent", "severity": "info", "note": "x"}],
        )
    )["record"]
    records.record_reconcile(
        {"reviews": [primary["event_id"], blind["event_id"], tiebreak["event_id"]]}
    )
    final = _hook("stop", workspace)
    assert final.returncode == 0, final.stdout
    status = _status(workspace)
    assert status["verdict"] == "pass"
    assert any("treat it as unknown" in w for w in status["warnings"])


def test_stop_requires_song_receipt(workspace: Path) -> None:
    assert _hook("session-start", workspace).returncode == 0
    _song(workspace)
    denied = _hook("stop", workspace)
    assert denied.returncode == 2
    assert "bard sent you a song" in denied.stdout
    records.record_song_receipt(
        {"delivery": "liaison/ode-to-w1.bard-song.json", "felt": FELT, "prompted_review": False}
    )
    assert _hook("stop", workspace).returncode == 0
    assert not (workspace / "observations" / "bard").exists()


def test_session_start_recalls_sister_concerns(workspace: Path) -> None:
    _connectivity(workspace)
    _sister_impression(workspace, "fpga", "fpga/board.connectivity.json", SISTER_TEXT)
    _song(workspace)
    result = _hook("session-start", workspace, session="s2")
    assert result.returncode == 0
    context = json.loads(result.stdout)["additionalContext"]
    assert "J1 pin 3 carries VBUS" in context
    assert "bard sent you a song" in context


def test_records_policy_matches_core() -> None:
    policy = HOOK_RECORDS.load_policy(SCRIPTS.parents[1])
    assert policy["plugin"] == records.PLUGIN
    assert policy["records_dir"] == records.RECORDS_DIR.as_posix()
    assert policy["schema_version"] == records.SCHEMA_VERSION == 2
    assert policy["dual_review_globs"]
    assert HOOK_RECORDS.IMPRESSION_MIN_CHARS == records.IMPRESSION_MIN_CHARS
    assert HOOK_RECORDS.LOG_FILES == records.LOG_FILES
    assert set(records.Severity.__args__) == HOOK_RECORDS.SEVERITIES
    assert set(records.Disposition.__args__) == HOOK_RECORDS.DISPOSITIONS
    hooks = json.loads((SCRIPTS.parent / "hooks.json").read_text(encoding="utf-8"))
    for event, mode in (("session_start", "session-start"), ("stop", "stop")):
        commands = [h["command"] for g in hooks[event] for h in g["hooks"]]
        assert any(f'require_records.py" {mode}' in c for c in commands)


# --- blind guard ---------------------------------------------------------------


def _guard(tool: str, tool_input: dict[str, Any]) -> int:
    return subprocess.run(
        [sys.executable, str(BLIND_GUARD)],
        input=json.dumps({"tool_name": tool, "tool_input": tool_input}),
        capture_output=True,
        text=True,
        check=False,
    ).returncode


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        (
            "file_editor",
            {"command": "view", "path": "/w/observations/fpga/vision-reviews.jsonl"},
        ),
        ("terminal", {"command": "cat observations/fpga/impressions.jsonl"}),
        ("terminal", {"command": "ls observations"}),
        ("grep", {"pattern": "W1", "path": "out/review-visual-harness-diagram.advisory.json"}),
        ("glob", {"pattern": "**/*.advisory.json"}),
        ("file_editor", {"command": "view", "path": "liaison/r1.ux-response.json"}),
    ],
)
def test_blind_guard_denies_earlier_reviews(tool: str, tool_input: dict[str, Any]) -> None:
    assert _guard(tool, tool_input) == 2


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("file_editor", {"command": "view", "path": "out/fpga-reports/demo.fpga-pinmap.png"}),
        ("terminal", {"command": "cat demo.contract.json"}),
        ("inspect_image_with_vision", {"question": "observations of the drawing?"}),
    ],
)
def test_blind_guard_allows_the_image_and_contract(tool: str, tool_input: dict[str, Any]) -> None:
    assert _guard(tool, tool_input) == 0
