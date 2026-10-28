#!/usr/bin/env python3
"""Unit tests for freeze_manifest.py artifact-integrity fixes.

Run:  python3 -m pytest analysis/test_freeze_manifest.py -q
 or:  python3 analysis/test_freeze_manifest.py      (no pytest needed)
"""
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "freeze_manifest.py"


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _run(tmp: Path, *args):
    pr = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp,
                        capture_output=True, text=True)
    assert pr.returncode == 0, pr.stderr
    return pr


def _mk(tmp: Path):
    rows = [{"file": "r1/a.json", "verdict": "TRIGGERED", "exec_cwe": "CWE-22",
             "executable": True, "triggered": True, "confirmed": True},
            {"file": "r1/b.json", "verdict": "NOT_TRIGGERED", "exec_cwe": None,
             "executable": True, "triggered": False, "confirmed": False}]
    (tmp / "census_pinned.json").write_text(json.dumps({"tier2": True, "n_candidates": 2, "rows": rows}))
    # a stale, differently-named census must NOT stand in for the one used
    (tmp / "corpus_census.json").write_text(json.dumps({"rows": []}))
    (tmp / "consortium_confirmed_P5.json").write_text("{}")
    root = tmp / "calibration" / "mined_prompts"
    for repo, n in (("r1", 2), ("r2", 0), ("ai-ssti-r1", 3), ("llama-cpp-ssti", 1)):
        (root / repo).mkdir(parents=True)
        for i in range(n):
            (root / repo / f"c{i}.json").write_text("{}")


def test_census_file_is_hashed_under_its_own_name():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _mk(tmp)
        _run(tmp, "--census", "census_pinned.json", "--out", "M.json")
        man = json.loads((tmp / "M.json").read_text())
        assert man["artifacts"]["census_pinned.json"]["sha256_16"] == _sha16(tmp / "census_pinned.json")
        assert man["census"]["sha256_16"] == _sha16(tmp / "census_pinned.json")
        assert man["census"]["funnel"]["confirmed"] == 1


def test_rq2_artifacts_are_hashed():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _mk(tmp)
        _run(tmp, "--census", "census_pinned.json", "--out", "M.json")
        man = json.loads((tmp / "M.json").read_text())
        assert "consortium_confirmed_P5.json" in man["artifacts"]


def test_md_sidecar_follows_out_and_never_clobbers_cwd_manifest():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _mk(tmp)
        (tmp / "MANIFEST.md").write_text("COMMITTED")
        (tmp / "scratch").mkdir()
        _run(tmp, "--census", "census_pinned.json", "--out", "scratch/M.json")
        assert (tmp / "scratch" / "M.md").exists()
        assert (tmp / "MANIFEST.md").read_text() == "COMMITTED"


def test_substudy_dirs_counted_separately():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _mk(tmp)
        _run(tmp, "--census", "census_pinned.json", "--out", "M.json")
        mc = json.loads((tmp / "M.json").read_text())["mined_corpus"]
        assert mc["n_repo_dirs"] == 4
        assert mc["n_repos_main_census"] == 2
        assert mc["n_repos_main_census_with_candidates"] == 1
        assert mc["excluded_substudy_dirs"] == ["ai-ssti-r1", "llama-cpp-ssti"]


def test_hand_curated_corpus_release_is_preserved():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _mk(tmp)
        (tmp / "M.json").write_text(json.dumps({"corpus_release": {"note": "keep me"}}))
        _run(tmp, "--census", "census_pinned.json", "--out", "M.json")
        assert json.loads((tmp / "M.json").read_text())["corpus_release"] == {"note": "keep me"}


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
    sys.exit(0 if passed == len(fns) else 1)
