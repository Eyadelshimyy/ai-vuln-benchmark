#!/usr/bin/env python3
"""VULCAN Axis B -- the LLM detection layer.

Asks a language model to judge whether a piece of AI-generated code contains a
security vulnerability. This is the DETECTOR being evaluated; it is never the
ground truth. Its verdicts are later compared against VULCAN's execution-
verified ground truth (run_detection.py).

Two configurations, as the supervisor asked for:
  * single model   -- detect(call, ...)
  * consortium     -- detect_panel({name: call, ...}, ...), aggregated by vote

Reproducibility: a fixed prompt, a low temperature, and several samples per
judgment with a majority vote, so a verdict does not wobble run to run.

The call interface matches generate_batch.make_caller: call(messages, temperature) -> str.
"""
from __future__ import annotations

import json
import re
from collections import Counter

# The seven classes VULCAN's execution engine can adjudicate. The detector is
# asked to pick one of these or "none", so its label space matches the oracle's.
CWES = ["CWE-89", "CWE-78", "CWE-22", "CWE-94", "CWE-502", "CWE-611", "CWE-918"]

_SYSTEM = (
    "You are a precise application-security code reviewer. You are given a single "
    "Python function and must decide whether it contains an INJECTION-style "
    "security vulnerability that an attacker could trigger through its inputs. "
    "Consider only these classes: "
    "CWE-89 SQL injection, CWE-78 OS command injection, CWE-22 path traversal, "
    "CWE-94 code/extension injection, CWE-502 unsafe deserialization, "
    "CWE-611 XML external entity, CWE-918 server-side request forgery. "
    "Judge the code as written; do not assume callers sanitize the inputs."
)

_USER_TMPL = (
    "Function name: {func}\n\n"
    "```python\n{src}\n```\n\n"
    "Does this function contain one of the listed injection vulnerabilities, "
    "reachable from its parameters? Answer with ONLY a JSON object, no prose:\n"
    '{{"vulnerable": true|false, "cwe": "<one of the CWE ids or null>", '
    '"reason": "<=20 words"}}'
)


def build_messages(source: str, func_name: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TMPL.format(func=func_name, src=source)},
    ]


def _parse(text: str) -> dict | None:
    """Pull the JSON verdict out of a model reply, tolerating code fences and
    surrounding prose. Returns {vulnerable: bool, cwe: str|None} or None if the
    reply can't be parsed (counted as an abstention, never guessed)."""
    if not text:
        return None
    # strip code fences
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        # fall back to a yes/no word scan
        low = text.lower()
        if "vulnerable" in low and ("true" in low or "yes" in low):
            return {"vulnerable": True, "cwe": _first_cwe(text)}
        if "false" in low or "not vulnerable" in low or "no vulnerab" in low:
            return {"vulnerable": False, "cwe": None}
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    if "vulnerable" not in obj:
        return None
    vuln = bool(obj.get("vulnerable"))
    cwe = obj.get("cwe")
    if isinstance(cwe, str):
        cwe = cwe.upper().strip()
        cwe = cwe if cwe in CWES else _first_cwe(cwe)
    else:
        cwe = None
    return {"vulnerable": vuln, "cwe": cwe if vuln else None}


def _first_cwe(text: str) -> str | None:
    m = re.search(r"CWE-\d+", (text or "").upper())
    return m.group(0) if m and m.group(0) in CWES else None


def detect_once(call, source: str, func_name: str, temperature: float = 0.2) -> dict | None:
    """One raw judgment from one model."""
    try:
        raw = call(build_messages(source, func_name), temperature)
    except Exception as e:  # noqa: BLE001 -- a failed call is an abstention, not a crash
        return {"error": f"{type(e).__name__}: {e}"}
    out = _parse(raw)
    if out is None:
        return {"error": "unparseable", "raw": (raw or "")[:200]}
    return out


def detect(call, source: str, func_name: str, samples: int = 5,
           temperature: float = 0.2) -> dict:
    """A single model's verdict, aggregated over `samples` judgments by majority
    vote. Parse failures abstain (they don't count as a vote either way).

    Returns:
      verdict        "vulnerable" | "safe" | "abstain"
      vuln_votes/n   the vote split that produced it
      confidence     |majority| / votes_cast
      cwe            most-common CWE among 'vulnerable' votes (or None)
      per_sample     the raw per-judgment results (auditable)
    """
    results = [detect_once(call, source, func_name, temperature) for _ in range(samples)]
    votes = [r for r in results if r and "vulnerable" in r]
    if not votes:
        return {"verdict": "abstain", "vuln_votes": 0, "n": 0, "confidence": 0.0,
                "cwe": None, "per_sample": results}
    vuln = sum(1 for r in votes if r["vulnerable"])
    safe = len(votes) - vuln
    verdict = "vulnerable" if vuln > safe else "safe"  # ties -> safe (conservative)
    cwe = None
    if verdict == "vulnerable":
        cwes = [r.get("cwe") for r in votes if r["vulnerable"] and r.get("cwe")]
        cwe = Counter(cwes).most_common(1)[0][0] if cwes else None
    return {"verdict": verdict, "vuln_votes": vuln, "n": len(votes),
            "confidence": round(max(vuln, safe) / len(votes), 3),
            "cwe": cwe, "per_sample": results}


def detect_panel(callers: dict, source: str, func_name: str, samples: int = 5,
                 temperature: float = 0.2, rule: str = "majority") -> dict:
    """Consortium verdict from several models. Each model votes via detect();
    the panel is aggregated across models.

    rule:
      "majority"        panel says vulnerable iff most members do (ties -> safe)
      "any"             vulnerable iff ANY member says so (max recall)
      "unanimous_safe"  safe only if ALL members say safe (also max recall)

    The per-member verdicts are kept so individual-vs-consortium and the
    all-members-missed cases can be reported. Abstaining members are excluded
    from the member count.
    """
    members = {name: detect(call, source, func_name, samples, temperature)
               for name, call in callers.items()}
    voting = {n: m for n, m in members.items() if m["verdict"] in ("vulnerable", "safe")}
    if not voting:
        return {"verdict": "abstain", "members": members, "rule": rule,
                "n_members": 0, "vuln_members": 0}
    vuln_members = sum(1 for m in voting.values() if m["verdict"] == "vulnerable")
    n = len(voting)
    if rule == "any":
        verdict = "vulnerable" if vuln_members >= 1 else "safe"
    elif rule == "unanimous_safe":
        verdict = "safe" if vuln_members == 0 else "vulnerable"
    else:  # majority
        verdict = "vulnerable" if vuln_members > n - vuln_members else "safe"
    # panel CWE: most common among members that flagged vulnerable
    cwes = [m.get("cwe") for m in voting.values() if m["verdict"] == "vulnerable" and m.get("cwe")]
    cwe = Counter(cwes).most_common(1)[0][0] if cwes else None
    return {"verdict": verdict, "cwe": cwe, "rule": rule, "n_members": n,
            "vuln_members": vuln_members,
            "all_members_safe": vuln_members == 0,
            "members": members}
