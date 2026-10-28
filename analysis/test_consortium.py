#!/usr/bin/env python3
"""Unit tests for detector_consortium.py integrity fixes.

Run:  python3 -m pytest analysis/test_consortium.py -q
 or:  python3 analysis/test_consortium.py      (no pytest needed)
"""
import json, importlib.util, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dc", HERE / "detector_consortium.py")
dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)


# ---- STRICT JSON parsing ----------------------------------------------------
def test_parse_true_bool():
    assert dc._parse('{"vulnerable": true, "cwe": "CWE-78"}')["vulnerable"] is True

def test_parse_false_bool():
    assert dc._parse('{"vulnerable": false}')["vulnerable"] is False

def test_parse_string_false_is_invalid():
    # "false" as a STRING must NOT become True (the old bug) -> INVALID
    assert dc._parse('{"vulnerable": "false"}')["vulnerable"] is None

def test_parse_missing_field_is_invalid():
    assert dc._parse('{"cwe": "CWE-22"}')["vulnerable"] is None

def test_parse_int_one_is_invalid():
    assert dc._parse('{"vulnerable": 1}')["vulnerable"] is None

def test_parse_non_json_is_invalid():
    assert dc._parse('I think this is vulnerable')["vulnerable"] is None

def test_parse_prose_wrapped_json_ok():
    assert dc._parse('sure: {"vulnerable": true} done')["vulnerable"] is True


# ---- STRICT label construction ---------------------------------------------
def _mk_root(tmp, files):
    root = Path(tmp)
    for name, src in files.items():
        (root / name).write_text(json.dumps({"ground_truth_source": src, "function": "f"}))
    return root

def test_build_eval_set_labels():
    with tempfile.TemporaryDirectory() as tmp:
        files = {"a.json": "def f(): pass", "b.json": "def f(): pass",
                 "c.json": "def f(): pass", "d.json": "def f(): pass"}
        root = _mk_root(tmp, files)
        rows = [
            {"file": "a.json", "verdict": "TRIGGERED",         "triggered": True,  "confirmed": True,  "cwes": ["CWE-22"]},
            {"file": "b.json", "verdict": "NOT_TRIGGERED",     "triggered": False, "confirmed": False, "cwes": ["CWE-22"]},
            {"file": "c.json", "verdict": "TRIGGERED",         "triggered": True,  "confirmed": False, "cwes": ["CWE-89"]},
            {"file": "d.json", "verdict": "COULD_NOT_EXECUTE", "triggered": False, "confirmed": False, "cwes": ["CWE-89"]},
        ]
        items = dc.build_eval_set(rows, root, "confirmed")
        by = {it["file"]: it["label"] for it in items}
        assert by.get("a.json") is True            # confirmed -> positive
        assert by.get("b.json") is False           # NOT_TRIGGERED -> clean negative
        assert "c.json" not in by                  # triggered-unconfirmed under 'confirmed' -> EXCLUDED
        assert "d.json" not in by                  # COULD_NOT_EXECUTE -> EXCLUDED

def test_build_eval_set_triggered_label():
    with tempfile.TemporaryDirectory() as tmp:
        root = _mk_root(tmp, {"c.json": "def f(): pass"})
        rows = [{"file": "c.json", "verdict": "TRIGGERED", "triggered": True, "confirmed": False, "cwes": []}]
        items = dc.build_eval_set(rows, root, "triggered")
        assert items and items[0]["label"] is True   # triggered label -> positive


# ---- experiment-aware cache key --------------------------------------------
def test_cache_key_changes_with_source_and_model():
    it1 = {"file": "x.json", "source": "AAA"}
    it2 = {"file": "x.json", "source": "BBB"}
    c = Path("/tmp")
    k_src1 = dc._cache_key(c, "m", "model-1", it1)
    k_src2 = dc._cache_key(c, "m", "model-1", it2)
    k_mdl  = dc._cache_key(c, "m", "model-2", it1)
    assert k_src1 != k_src2          # source change invalidates
    assert k_src1 != k_mdl           # model-id change invalidates

def test_cache_key_changes_with_prompt_version(monkeypatch=None):
    it = {"file": "x.json", "source": "AAA"}
    k1 = dc._cache_key(Path("/tmp"), "m", "model-1", it)
    old = dc.PROMPT_VERSION
    try:
        dc.PROMPT_VERSION = old + "-changed"
        k2 = dc._cache_key(Path("/tmp"), "m", "model-1", it)
    finally:
        dc.PROMPT_VERSION = old
    assert k1 != k2                  # prompt-version change invalidates


# ---- scoring ----------------------------------------------------------------
def test_score_counts():
    items = [{"file": "p1", "label": True, "cwes": ["CWE-22"]},
             {"file": "p2", "label": True, "cwes": ["CWE-22"]},
             {"file": "n1", "label": False, "cwes": []},
             {"file": "n2", "label": False, "cwes": []}]
    preds = {"p1": True, "p2": False, "n1": True, "n2": False}   # 1TP 1FN 1FP 1TN
    m = dc.score(preds, items)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 1)
    assert abs(m["precision"] - 0.5) < 1e-9 and abs(m["recall"] - 0.5) < 1e-9
    assert m["per_cwe"]["CWE-22"] == {"num": 1, "den": 2}   # recall denom = positives only


# ---- F1 bootstrap CI --------------------------------------------------------
def test_bootstrap_f1_ci_bounds():
    pairs = [(True, True)] * 8 + [(True, False)] * 2 + [(False, False)] * 10
    point, lo, hi = dc.bootstrap_f1_ci(pairs, n_boot=500, seed=0)
    assert point is not None and 0.0 <= lo <= point <= hi <= 1.0


# ---- consortium majority on common set (via a tiny end-to-end simulation) --
def test_majority_vote_common_only():
    # three models; file 'c' answered by only two -> excluded from common
    items = [{"file": "a", "label": True, "cwes": []},
             {"file": "b", "label": False, "cwes": []},
             {"file": "c", "label": True, "cwes": []}]
    mp = {"m1": {"a": True, "b": False, "c": True},
          "m2": {"a": True, "b": True, "c": True},
          "m3": {"a": False, "b": False}}            # m3 missing 'c'
    real = ["m1", "m2", "m3"]
    common = set.intersection(*[set(mp[n]) for n in real])
    assert common == {"a", "b"}                      # 'c' dropped (m3 missing)
    items_common = [it for it in items if it["file"] in common]
    consensus = {}
    for it in items_common:
        votes = [mp[m][it["file"]] for m in real]
        consensus[it["file"]] = sum(votes) > len(votes) / 2
    assert consensus == {"a": True, "b": False}      # a: 2/3 yes, b: 1/3 yes


# ---- publication-grade statistics ------------------------------------------
def test_precision_at_prevalence():
    # recall 0.7, fpr 0.2: 33% -> ~0.64, 6% -> ~0.18 (precision drops as vulns get rarer)
    assert abs(dc.precision_at(0.7, 0.2, 0.333) - 0.636) < 0.02
    p6 = dc.precision_at(0.7, 0.2, 0.06)
    assert p6 < 0.25 and p6 < dc.precision_at(0.7, 0.2, 0.333)

def test_cohens_kappa_identical_and_chance():
    files = ["a", "b", "c", "d"]
    ident = ({f: True for f in files}, {f: True for f in files})
    # all-True vs all-True: perfect agreement but zero variance -> kappa defined as 1.0
    assert dc.cohens_kappa(ident[0], ident[1], files) == 1.0
    a = {"a": True, "b": True, "c": False, "d": False}
    assert abs(dc.cohens_kappa(a, a, files) - 1.0) < 1e-9          # identical -> 1
    b = {"a": False, "b": False, "c": True, "d": True}
    assert dc.cohens_kappa(a, b, files) < 0                        # opposite -> negative

def test_mcnemar_detects_difference():
    # A correct everywhere, B wrong on 6 that A got right -> strongly significant
    ca = [True] * 10
    cb = [True] * 4 + [False] * 6
    r = dc.mcnemar_exact(ca, cb)
    assert r["b10"] == 6 and r["b01"] == 0 and r["p"] < 0.05
    # identical -> not significant
    assert dc.mcnemar_exact(ca, ca)["p"] == 1.0

def test_bootstrap_f1_diff_sign():
    # A clearly better than B -> positive point diff
    a = [(True, True)] * 8 + [(True, False)] * 2 + [(False, False)] * 10
    b = [(True, False)] * 10 + [(False, True)] * 5 + [(False, False)] * 5
    point, lo, hi = dc.bootstrap_f1_diff_ci(a, b, n_boot=500, seed=0)
    assert point > 0 and lo <= point <= hi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    import sys
    sys.exit(0 if passed == len(fns) else 1)   # non-zero on failure so scripts/CI notice
