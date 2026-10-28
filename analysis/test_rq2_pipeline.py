#!/usr/bin/env python3
"""Tests for the RQ2 run preparation (label set, spend guard, protocols) and the
CWE-22 subtype stratifier. No model is ever called: the call adapters are
replaced with functions that fail the test if reached.

Run:  python3 -m pytest analysis/test_rq2_pipeline.py -q
 or:  python3 analysis/test_rq2_pipeline.py      (no pytest needed)
"""
import contextlib, hashlib, importlib.util, io, json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


dc = _load("detector_consortium")
rl = _load("rq2_labelset")
cs = _load("cwe22_subtype")


def _boom(*a, **k):
    raise AssertionError("a real model call was attempted")


_real_call_openai = dc._call_openai
dc._call_opencode_cli = _boom
dc._call_openai = _boom
CFG = {"name": "m1", "cli": True, "model": "x/y"}


# ---- synthetic corpus ---------------------------------------------------------
def _mk_corpus(tmp: Path):
    root = tmp / "calibration" / "mined_prompts"
    (root / "r1").mkdir(parents=True)
    srcs = {
        "pos_a.json": ("read", "def read(path):\n    return open(path).read()\n", "cursor"),
        "pos_b.json": ("save", "def save(name):\n    open(os.path.join('/data', name), 'w')\n", "devin"),
        "pos_dup.json": ("read", "def read(path):\n    return open(path).read()\n", "cursor"),
    }
    for i in range(8):
        srcs[f"neg_{i}.json"] = (f"f{i}", f"def f{i}(x):\n    return x + {i}\n", "cursor" if i % 2 else "devin")
    srcs["trig.json"] = ("t", "def t(x):\n    return x\n", "devin")
    rows = []
    for fn, (func, src, tool) in sorted(srcs.items()):
        (root / "r1" / fn).write_text(json.dumps({
            "repo": "r1", "function": func, "ground_truth_source": src,
            "ai_provenance": {"ai_tools": {tool: 1}}}))
        pos = fn.startswith("pos")
        rows.append({"file": f"r1/{fn}", "cwes": ["CWE-22"] if pos else [],
                     "verdict": "TRIGGERED" if (pos or fn == "trig.json") else "NOT_TRIGGERED",
                     "exec_cwe": "CWE-22" if pos else None, "confirmed_cwe": "CWE-22" if pos else None,
                     "executable": True, "triggered": pos or fn == "trig.json", "confirmed": pos})
    census = tmp / "census.json"
    census.write_text(json.dumps({"tier2": True, "n_candidates": len(rows), "rows": rows}))
    return census, root


@contextlib.contextmanager
def _in(tmp):
    old = os.getcwd(); os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)


def _run_main(argv):
    old = sys.argv; sys.argv = ["detector_consortium.py", *argv]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            dc.main()
    finally:
        sys.argv = old
    return err.getvalue()


# ---- label set ----------------------------------------------------------------
def test_labelset_dedups_sources_and_excludes_inconclusive():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        ls = rl.build(census, root, ratio=2.0)
        m = ls["meta"]
        assert m["n_pos"] == 2, m                      # pos_a and pos_dup are one function
        assert m["population"]["excluded_rows"] == {"TRIGGERED_unconfirmed": 1}
        assert all(not it["label"] for it in ls["items"] if it["file"].startswith("r1/neg"))
        # negatives drawn per generator in proportion to positives (1 cursor, 1 devin -> 2 each)
        assert m["strata"]["neg_by_generator"] == {"cursor": 2, "devin": 2}, m["strata"]
        dup = [it for it in ls["items"] if it["file"] == "r1/pos_a.json"][0]
        assert dup["duplicate_files"] == ["r1/pos_dup.json"]


def test_labelset_is_deterministic():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        assert rl.build(census, root, 2.0)["meta"]["items_sha256"] == rl.build(census, root, 2.0)["meta"]["items_sha256"]


def _write_labelset(tmp, census, root):
    ls = rl.build(census, root, 2.0)
    ls["meta"]["root"] = str(root)
    p = tmp / "ls.json"; p.write_text(json.dumps(ls)); return p


def test_labelset_detects_changed_source():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        p = _write_labelset(tmp, census, root)
        rec = json.loads((root / "r1" / "pos_a.json").read_text())
        rec["ground_truth_source"] += "# edited\n"
        (root / "r1" / "pos_a.json").write_text(json.dumps(rec))
        try:
            dc.load_labelset(p)
        except SystemExit as e:
            assert "changed" in str(e)
        else:
            raise AssertionError("tampered source not detected")


# ---- spend guard ---------------------------------------------------------------
def test_no_live_flag_means_no_call_and_no_output():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        ls = _write_labelset(tmp, census, root)
        (tmp / "models.json").write_text(json.dumps([{"name": "m1", "cli": True, "model": "x/y"}]))
        with _in(tmp):
            err = _run_main(["--models", "models.json", "--labelset", str(ls), "--census", str(census),
                             "--protocol", "v2", "--cache", "cache", "--out", "out.json"])
        assert "NOT CALLING" in err, err
        assert not (tmp / "out.json").exists()


def test_smoke_requires_live():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        ls = _write_labelset(tmp, census, root)
        (tmp / "models.json").write_text(json.dumps([{"name": "m1", "cli": True, "model": "x/y"}]))
        with _in(tmp):
            try:
                _run_main(["--models", "models.json", "--labelset", str(ls), "--census", str(census),
                           "--smoke", "2", "--out", "out.json"])
            except SystemExit as e:
                assert "--live" in str(e)
            else:
                raise AssertionError("smoke ran without --live")


def test_fully_cached_run_needs_no_live_and_scores():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        lsp = _write_labelset(tmp, census, root)
        items, _ = dc.load_labelset(lsp)
        cache = tmp / "cache"; cache.mkdir()
        for it in items:   # pre-seed answers: flag exactly the positives
            ck = dc._cache_key(cache, "m1", "x/y", it, "v2", CFG)
            ck.write_text(json.dumps({"vulnerable": it["label"], "cwe": None}))
        (tmp / "models.json").write_text(json.dumps([{"name": "m1", "cli": True, "model": "x/y"}]))
        with _in(tmp):
            _run_main(["--models", "models.json", "--labelset", str(lsp), "--census", str(census),
                       "--protocol", "v2", "--cache", "cache", "--out", "out.json"])
        out = json.loads((tmp / "out.json").read_text())
        assert out["models"]["m1"]["recall"] == 1.0 and out["models"]["m1"]["fpr"] == 0.0
        assert out["protocol"] == "v2" and out["labelset"]["items_sha256"]
        assert out["per_stratum"]["generator"]["cursor"]["m1"]["n_pos"] == 1
        assert set(out["predictions"]["m1"]) == {it["file"] for it in items}


def test_framing_comparison_is_paired_and_counts_discordance():
    rc = _load("rq2_compare")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); census, root = _mk_corpus(tmp)
        lsp = _write_labelset(tmp, census, root)
        items, _ = dc.load_labelset(lsp)
        cache = tmp / "cache"; cache.mkdir()
        for it in items:
            # A (v2-threat) flags every positive; B (v2) flags none -> all positives discordant A-only
            dc._cache_key(cache, "m1", "x/y", it, "v2-threat", CFG).write_text(json.dumps({"vulnerable": it["label"]}))
            dc._cache_key(cache, "m1", "x/y", it, "v2", CFG).write_text(json.dumps({"vulnerable": False}))
        (tmp / "models.json").write_text(json.dumps([{"name": "m1", "cli": True, "model": "x/y"}]))
        with _in(tmp):
            for proto, out in (("v2-threat", "a.json"), ("v2", "b.json")):
                _run_main(["--models", "models.json", "--labelset", str(lsp), "--census", str(census),
                           "--protocol", proto, "--cache", "cache", "--out", out])
        res = rc.compare(json.loads((tmp / "a.json").read_text()), json.loads((tmp / "b.json").read_text()),
                         json.loads(lsp.read_text())["items"])
        pos = res["m1"]["positives (all)"]
        assert (pos["n"], pos["A_only"], pos["B_only"]) == (2, 2, 0), pos
        assert res["m1"]["negatives (all)"]["A_only"] == 0
        assert "len_tercile" in json.loads((tmp / "a.json").read_text())["per_stratum"]


# ---- OpenRouter adapter (fake client; nothing leaves the process) ----------------------
class _FakeResp:
    def __init__(self, content, finish="stop"):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg, "finish_reason": finish})()]
        self.usage = type("U", (), {"total_tokens": 50})()
        self._d = {"id": "gen-1", "model": "x/y-0101", "provider": "ProvA",
                   "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50, "cost": 0.0012,
                             "completion_tokens_details": {"reasoning_tokens": 7}}}

    def model_dump(self):
        return self._d


def _with_fake_openai(content, finish="stop"):
    import types
    sent = {}

    class _Client:
        def __init__(self, api_key=None, base_url=None):
            sent["api_key"] = api_key; sent["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kw):
            sent["kw"] = kw
            return _FakeResp(content, finish)

    fake = types.ModuleType("openai"); fake.OpenAI = _Client
    return fake, sent


def _call_with_fake(cfg, content, finish="stop", env=None):
    fake, sent = _with_fake_openai(content, finish)
    old_mod = sys.modules.get("openai"); sys.modules["openai"] = fake
    old_env = dict(os.environ); os.environ.update(env or {})
    try:
        ans = _real_call_openai(cfg, "def f(p): open(p)", "f", 10, "v2-threat")
    finally:
        sys.modules.pop("openai")
        if old_mod is not None:
            sys.modules["openai"] = old_mod
        os.environ.clear(); os.environ.update(old_env)
    return ans, sent


OR = {"name": "m", "model": "x/y", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "TEST_OR_KEY",
      "temperature": None, "seed": None, "max_tokens": 4000, "reasoning_effort": "low", "provider": {"order": ["p"]}}


def test_openrouter_request_respects_model_capabilities():
    ans, sent = _call_with_fake(OR, '{"vulnerable": true, "cwe": "CWE-22", "confidence": 0.9, "reason": "r"}',
                                env={"TEST_OR_KEY": "sk-secret-123"})
    kw = sent["kw"]
    assert "temperature" not in kw and "seed" not in kw and kw["max_tokens"] == 4000
    assert kw["extra_body"] == {"usage": {"include": True}, "reasoning": {"effort": "low"}, "provider": {"order": ["p"]}}
    assert ans["vulnerable"] is True and ans["served_model"] == "x/y-0101" and ans["served_provider"] == "ProvA"
    assert ans["usage"]["cost"] == 0.0012 and ans["reasoning_tokens"] == 7
    assert "sk-secret-123" not in json.dumps(ans)          # the key never reaches the cache


def test_openrouter_missing_key_sends_nothing():
    ans, sent = _call_with_fake(OR, "{}", env={})
    assert "no API key" in ans["error"] and "kw" not in sent


def test_reasoning_exhausted_cap_is_an_error_not_an_answer():
    ans, _ = _call_with_fake(OR, "", finish="length", env={"TEST_OR_KEY": "k"})
    assert "max_tokens" in ans["error"] and ans["vulnerable"] is None


def test_settings_change_invalidates_v2_cache_but_not_v1():
    it = {"file": "r/f.json", "source": "x"}
    a = dc._cache_key(Path("c"), "m", "x/y", it, "v2-threat", {**OR, "reasoning_effort": "low"})
    b = dc._cache_key(Path("c"), "m", "x/y", it, "v2-threat", {**OR, "reasoning_effort": "high"})
    assert a != b
    assert dc._cache_key(Path("c"), "m", "x/y", it, "v1", OR) == dc._cache_key(Path("c"), "m", "x/y", it, "v1")


def test_cost_estimate_uses_pricing():
    items = [{"source": "x" * 350}]
    lines = dc.estimate_cost([{"name": "m", "model": "x/y"}], items, {"m": 10},
                             {"data": [{"id": "x/y", "pricing": {"prompt": "0.000001", "completion": "0.00001"}}]},
                             [100, 1000], "v2-threat")
    assert lines[-1].startswith("cost est. TOTAL $") and "no pricing" not in lines[0]


# ---- cache semantics ---------------------------------------------------------------
def test_v1_cache_key_is_legacy_and_protocols_do_not_collide():
    it = {"file": "r/f.json", "source": "def f(): pass"}
    src_h = hashlib.md5(it["source"].encode()).hexdigest()
    legacy = hashlib.md5("\x00".join(["m", "id", dc.PROMPT_VERSION, it["file"], src_h]).encode()).hexdigest()
    assert dc._cache_key(Path("c"), "m", "id", it, "v1").name == f"m__{legacy}.json"
    keys = {dc._cache_key(Path("c"), "m", "id", it, p).name for p in dc.PROTOCOLS}
    assert len(keys) == len(dc.PROTOCOLS)


def test_cached_error_only_entry_is_retried():
    with tempfile.TemporaryDirectory() as t:
        ck = Path(t) / "k.json"
        ck.write_text(json.dumps({"error": "cli timeout", "vulnerable": None}))
        assert dc._cached_answer(ck) is None
        ck.write_text(json.dumps({"vulnerable": False, "cwe": None}))
        assert dc._cached_answer(ck) == {"vulnerable": False, "cwe": None}


def test_legacy_p5_cache_still_resolves():
    """Regression on the real P5 cache, if present: the v1 key of every P5 item
    still points at an existing answer (so P5 reproduces without calls)."""
    root = HERE.parent
    cache = root / ".consortium_cache"
    census = root / "corpus_census_P5_final.json"
    if not (cache.exists() and census.exists()):
        return
    rows = json.loads(census.read_text())["rows"]
    items = [it for it in dc.build_eval_set(rows, root / "calibration" / "mined_prompts", "confirmed") if it["label"]]
    missing = [it["file"] for it in items
               if not dc._cache_key(cache, "kimi-k3", "opencode-go/kimi-k3", it, "v1").exists()]
    assert not missing, missing[:3]


# ---- CWE-22 subtype -------------------------------------------------------------------
def test_cwe22_subtypes():
    C = cs.classify_source
    assert C("def f(path):\n    return open(path).read()\n")["subtype"] == cs.DIRECT
    assert C("def f(name):\n    return open(os.path.join(BASE, name))\n")["subtype"] == cs.BASE_JOIN
    assert C("def f(name):\n    p = Path('/srv') / name\n    return p.read_text()\n")["subtype"] == cs.BASE_JOIN
    assert C("def f(name):\n    return open(f'/data/{name}')\n")["subtype"] == cs.BASE_JOIN
    assert C("def f(name):\n    return open(ROOT + '/' + name)\n")["subtype"] == cs.BASE_JOIN
    # message text is not a path composition
    assert C("def f(path):\n    raise ValueError(f'file {path} missing')\n")["subtype"] == cs.DIRECT
    assert C("def f(path):\n    print('Error: ' + path)\n    open(path)\n")["subtype"] == cs.DIRECT
    # caller-supplied base directory: the caller controls the whole location
    assert C("def f(d):\n    return open(os.path.join(d, 'x.json'))\n")["subtype"] == cs.DIRECT
    # indented method source (as mined) parses
    assert C("    def m(self, path):\n        return open(path)\n", "m")["subtype"] == cs.DIRECT
    assert C("def broken(:\n")["subtype"] == cs.UNKNOWN


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
