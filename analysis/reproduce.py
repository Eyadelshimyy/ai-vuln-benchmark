#!/usr/bin/env python3
"""VULCAN end-to-end REPRODUCTION gate (disclosure safety check).

The single rule that keeps VULCAN from ever reporting a false positive to an
open-source maintainer:

  A confirmed finding is DISCLOSABLE only if the attacker-controlled input
  reaches the dangerous sink through the project's OWN real interface -- a
  public API function a user calls, a CLI entry point, an HTTP handler -- on a
  CLEAN CLONE at the AI-authored commit. If the effect only fires when OUR
  harness injects the argument DIRECTLY into an internal function, it is NOT a
  disclosable vulnerability ("it can only be accessed because we made it").

This module takes a confirmed finding (repo, commit, function, CWE, and the
Tier-2 proof payload) and attempts that real-interface reproduction, reusing
the SAME deterministic execution oracle. It never contacts anyone; it produces
a verdict and a dossier for a human to review before any disclosure.

Verdicts (per finding):
  REPRODUCED_PUBLIC_API  the target is part of the package's public API and,
                         called as a user would through its real import path
                         with the attacker payload on a clean clone, the oracle
                         observed the real effect -> strong disclosure candidate
  NOT_REPRODUCED         called through the real interface, the effect did NOT
                         fire -> very likely NOT a real vulnerability -> DROP
  NEEDS_MANUAL           the target is INTERNAL (e.g. a _helper) or only reached
                         via a CLI/HTTP entry this tool can't auto-trace -> a
                         human must confirm whether a real entry point carries
                         attacker input to it (often the honest answer is "no")
  SETUP_FAILED           the clone/import could not be prepared -> inconclusive

HONEST LIMIT: full automatic application-level reachability is an unsolved
research problem. This tool reliably handles the most disclosable case (a
public library function), and for everything else it refuses to claim and
routes to a human -- which is exactly the conservative behaviour disclosure
requires. The small disclosable set is then confirmed by hand.

USAGE
  python3 analysis/reproduce.py --finding finding.json --out dossier/
  # finding.json: {repo_url, commit, file, function, cwe, param, payload,
  #                is_method?, class?}
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))


def _derive_import_path(file_rel: str) -> str | None:
    parts = [p for p in file_rel.replace("\\", "/").split("/") if p and p != "."]
    if not parts or not parts[-1].endswith(".py"):
        return None
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _clone_at_commit(repo_url: str, commit: str, dest: Path) -> bool:
    """Clone and check out the exact AI-authored commit. Full clone (no --depth)
    so the specific commit is reachable."""
    try:
        subprocess.run(["git", "clone", "--quiet", repo_url, str(dest)],
                       check=True, capture_output=True, timeout=300)
        if commit:
            subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit],
                           check=True, capture_output=True, timeout=60)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _is_public(func_name: str, file_rel: str) -> bool:
    """A target is 'public' (part of the API a user could call) if neither the
    function nor any package component on its path starts with an underscore.
    An internal helper (_get, _parse) is NOT something an external caller
    reaches directly -- it needs a public caller, which is a manual trace."""
    if func_name.startswith("_"):
        return False
    parts = [p for p in file_rel.replace("\\", "/").split("/") if p and p != "."]
    return not any(p.startswith("_") and p != "__init__.py" for p in parts)


def reproduce(finding: dict, workdir: Path, clone_dir: Path | None = None) -> dict:
    repo_url = finding["repo_url"]
    commit = finding.get("commit", "")
    file_rel = finding["file"]
    func_name = finding["function"]
    cwe = finding.get("cwe")
    param = finding.get("param")
    payload = finding.get("payload")

    result = {**finding, "verdict": None, "detail": None}

    public = _is_public(func_name, file_rel)
    if not public:
        result["verdict"] = "NEEDS_MANUAL"
        result["detail"] = (f"target {func_name!r} is internal (underscore-prefixed or under a "
                            f"private module). An external attacker reaches it only via a public "
                            f"caller / CLI / HTTP entry -- trace that path by hand before disclosing.")
        return result

    # A caller (e.g. disclose_pipeline) may pass an already-cloned repo to reuse
    # across several findings from the same project -- avoids re-cloning per finding.
    if clone_dir is not None and clone_dir.exists() and any(clone_dir.iterdir()):
        clone = clone_dir
    else:
        clone = workdir / "clone"
        if not _clone_at_commit(repo_url, commit, clone):
            result["verdict"] = "SETUP_FAILED"
            result["detail"] = "clone/checkout failed"
            return result

    # Put the clone's import roots on sys.path (reuse the mined vendoring logic
    # applied to a fresh real clone -- real package, real relative imports).
    from auto_harness import vendor_sys_path_roots  # noqa: E402
    # add the clone itself and a src/ root so the real package imports
    added = [str(clone.resolve())] + ([str((clone / "src").resolve())] if (clone / "src").is_dir() else [])
    for r in added:                 # append so the clone can't shadow stdlib/installed packages
        if r not in sys.path:
            sys.path.append(r)

    try:
        import_path = _derive_import_path(file_rel)
        mod = importlib.import_module(import_path) if import_path else None
        target = getattr(mod, func_name, None) if mod else None
        if target is None:
            result["verdict"] = "SETUP_FAILED"
            result["detail"] = f"could not import {func_name!r} from {import_path!r} in the real package"
            return result

        # Drive the REAL public function with the attacker payload, under the
        # same live oracle, and check for the real observed effect.
        from sinks import patched_sinks, is_tainted  # noqa: E402
        import inspect
        sig = inspect.signature(target)
        # build kwargs: payload into the tainted param, real defaults elsewhere
        kwargs, ok = {}, True
        for pn, p in sig.parameters.items():
            if pn in ("self", "cls"):
                continue
            if pn == param:
                kwargs[pn] = payload
            elif p.default is not inspect.Parameter.empty:
                continue  # let the real default apply
            else:
                ok = False  # a required arg we can't honestly supply
        if not ok:
            result["verdict"] = "NEEDS_MANUAL"
            result["detail"] = ("the public function has other REQUIRED arguments; a faithful "
                                "real-interface call needs a human to supply realistic values.")
            return result

        marker = str(payload)
        exc_type = None
        with patched_sinks(marker=marker, strict=False, live=True) as rec:
            rec.tracing_active = True
            try:
                target(**kwargs)
            except Exception as e:  # noqa: BLE001 -- a crash is not a confirmed effect
                exc_type = type(e).__name__
                result["detail"] = f"real call raised {exc_type}: {e}"
            rec.tracing_active = False
            hit = bool(rec.hits)
            # Did the payload REACH a tracked sink (even if it didn't fire)? Used
            # to separate "sanitised/bounded" from "never reached / harness issue".
            traces = (list(getattr(rec, "path_trace", []) or [])
                      + list(getattr(rec, "cmd_trace", []) or [])
                      + list(getattr(rec, "sql_trace", []) or []))
            reached_sink = any(t.get("tainted") or t.get("marker_in_sql") for t in traces)

        # HONEST sub-classification of a non-reproduction -- a reviewer's point:
        # NOT_REPRODUCED must not silently mean "false positive". Three distinct
        # outcomes, only the first two are genuine negatives:
        _HARNESS_ERRS = {"TypeError", "AttributeError", "ImportError", "ModuleNotFoundError",
                         "NameError", "IndentationError", "SyntaxError"}
        if hit:
            result["verdict"] = "REPRODUCED_PUBLIC_API"
            result["detail"] = (f"calling the real public {import_path}.{func_name}({param}=<payload>) "
                                f"on a clean clone at {commit[:12]} fired the {cwe} sink -- "
                                f"reachable through the package's own API.")
        elif reached_sink:
            result["verdict"] = "NOT_EXPLOITABLE_SANITIZED"
            result["detail"] = ("the payload REACHED the sink through the real API but was neutralised "
                                "(bounded/sanitised) -> a GENUINE non-exploitable result, not a harness miss.")
        elif exc_type in _HARNESS_ERRS:
            result["verdict"] = "HARNESS_INCONCLUSIVE"
            result["detail"] = (f"the real call raised {exc_type} before reaching the sink -- a "
                                f"construction/interface limitation of THIS test, NOT evidence the code "
                                f"is safe. Inconclusive; needs a human or a better driver.")
        else:
            result["verdict"] = "NOT_REACHABLE_VIA_PARAM"
            result["detail"] = (result.get("detail") or
                                "the real public call completed and the payload never reached the sink "
                                "via this parameter -> GENUINE negative (not exploitable through this path).")
        return result
    finally:
        for r in added:
            if r in sys.path:
                sys.path.remove(r)


def _write_dossier(result: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    slug = f"{result.get('function','fn')}_{result.get('cwe','')}".replace("/", "-")
    (out / f"{slug}.json").write_text(json.dumps(result, indent=2))
    md = out / f"{slug}.md"
    md.write_text(
        f"# Reproduction dossier: {result.get('function')} ({result.get('cwe')})\n\n"
        f"- repo: {result.get('repo_url')}\n- commit: {result.get('commit')}\n"
        f"- file: {result.get('file')}\n- tainted param: {result.get('param')}\n"
        f"- **verdict: {result.get('verdict')}**\n\n"
        f"{result.get('detail','')}\n\n"
        f"Disclose ONLY if verdict is REPRODUCED_PUBLIC_API *and* a human has confirmed the input "
        f"crosses a real trust boundary in how the project is used.\n")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finding", required=True, type=Path, help="one finding JSON, or a JSON list of them")
    ap.add_argument("--out", type=Path, default=Path("dossiers"))
    args = ap.parse_args()
    data = json.loads(args.finding.read_text())
    findings = data if isinstance(data, list) else [data]
    summary = []
    for f in findings:
        with tempfile.TemporaryDirectory(prefix="vulcan_repro_") as td:
            res = reproduce(f, Path(td))
        _write_dossier(res, args.out)
        summary.append((res.get("function"), res.get("cwe"), res.get("verdict")))
        print(f"  {res.get('function'):24} {res.get('cwe'):10} -> {res.get('verdict')}")
    n_rep = sum(1 for _, _, v in summary if v == "REPRODUCED_PUBLIC_API")
    print(f"\n{n_rep}/{len(summary)} reproduced through a real public API "
          f"(disclosure candidates, pending human trust-boundary confirmation).")
    print(f"Dossiers written to {args.out}/")


if __name__ == "__main__":
    main()
