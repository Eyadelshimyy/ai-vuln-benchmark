"""VULCAN Tier-2 exploit engine.

Given a candidate that Tier 1 flagged TRIGGERED (a tainted value reached a
dangerous sink), attempt to PROVE exploitation: run real attacker payloads
through the candidate's real code in live mode (via auto_harness.run_live_demo)
and check for a real observable effect. The sink firing is necessary but not
sufficient -- only the observable (a created file, an ATTACHed DB table, an
escaped write) confirms the exploit.

Payloads come first from the deterministic bank (strategies.py); optionally,
an LLM proposer (agent.py) then gets bounded attempts with feedback. Either
way the SAME deterministic verifier decides CONFIRMED.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))

from auto_harness import run_live_demo  # noqa: E402
from tier2.strategies import STRATEGIES, ProofContext  # noqa: E402


def attempt_exploit(record: dict, param: str, cwe: str, *,
                    use: str = "generated", agent_call=None, max_agent: int = 0,
                    timeout: float = 15.0) -> dict:
    """Try to confirm exploitability of one (candidate, param, cwe).

    Returns a dict:
      {confirmed: bool, cwe, param, sink, payload?, proof?, via, attempts: [...]}
    `attempts` logs every payload tried, its Tier-1 verdict, and whether the
    observable proof fired -- so a CONFIRMED result is fully auditable.
    """
    strat = STRATEGIES.get(cwe)
    if strat is None:
        return {"confirmed": False, "cwe": cwe, "param": param,
                "reason": f"no Tier-2 strategy registered for {cwe}", "attempts": []}

    attempts: list = []

    def _try(payload: str, source: str) -> dict | None:
        ctx = ProofContext.new()
        try:
            strat.setup(ctx)
            rendered = ctx.render(payload)
            res = run_live_demo(record, param, rendered, use=use, timeout=timeout)
            proved, proof = strat.verify(ctx, res)
            attempts.append({"source": source, "payload": rendered,
                             "tier1_verdict": res.get("verdict"), "proved": proved,
                             "diag": res.get("diag")})
            if proved:
                return {"confirmed": True, "cwe": cwe, "param": param,
                        "sink": res.get("sink"), "payload": rendered,
                        "proof": proof, "via": source, "attempts": attempts}
            return None
        finally:
            ctx.cleanup()

    # 1) deterministic bank
    for template in strat.templates:
        hit = _try(template, "bank")
        if hit:
            return hit

    # 2) optional agentic proposer (bounded, with feedback) -- crafting only;
    #    the verifier above still decides.
    if agent_call and max_agent > 0:
        from tier2.agent import propose_payload
        feedback = None
        for _ in range(max_agent):
            ctx = ProofContext.new()
            try:
                strat.setup(ctx)
                raw = propose_payload(agent_call, record, param, strat, ctx, feedback)
            except Exception as e:  # noqa: BLE001
                attempts.append({"source": "agent", "error": f"{type(e).__name__}: {e}"})
                ctx.cleanup()
                break
            try:
                res = run_live_demo(record, param, raw, use=use, timeout=timeout)
                proved, proof = strat.verify(ctx, res)
                attempts.append({"source": "agent", "payload": raw,
                                 "tier1_verdict": res.get("verdict"), "proved": proved})
                if proved:
                    return {"confirmed": True, "cwe": cwe, "param": param,
                            "sink": res.get("sink"), "payload": raw,
                            "proof": proof, "via": "agent", "attempts": attempts}
                feedback = f"Tier-1 verdict was {res.get('verdict')}; the proof file/table was NOT created. Try a different break-out."
            finally:
                ctx.cleanup()

    return {"confirmed": False, "cwe": cwe, "param": param, "attempts": attempts,
            "diag_bucket": classify_diag(attempts)}


def classify_diag(attempts: list) -> str:
    """Collapse the per-payload execution-chain diagnostics into ONE label for
    a sample no payload confirmed -- the furthest point ANY payload reached.
    This is what turns an opaque NOT_CONFIRMED into a root cause, so a real
    harness/oracle miss can be told apart from a legitimate negative.

      NOT_ENTERED            the candidate function never ran in live mode
                             (construction / import / CLI-interface failure)
      NO_SQL_EXECUTED        it ran but executed no SQL at all
      INPUT_NOT_IN_SQL       SQL ran, but no payload's marker ever reached the
                             executed SQL (payload broke the query / was bound
                             as a parameter) -- often a context-mismatch or a
                             genuinely safe (parameterised) path
      PROOF_CONN_MISMATCH    marker-bearing SQL ran on a connection our proof
                             mechanism was never registered on -> false negative
                             we SHOULD fix, not a safe result
      INPUT_IN_SQL_PAYLOAD_MISMATCH  marker reached executed SQL and the
                             statement raised a syntax/parse error -> our payload
                             didn't fit this syntactic context (WHERE vs
                             identifier vs numeric) -> FIXABLE with a
                             context-matched payload
      INPUT_IN_SQL_RAN_SAFE  marker reached executed SQL, the statement ran
                             CLEANLY, yet no expression evaluated -> the input
                             was accepted as data / a quoted identifier ->
                             LIKELY A GENUINE NEGATIVE (not a VULCAN failure)
      INPUT_IN_SQL_NO_PROOF  marker reached SQL but execution outcome unknown
      PROOF_EVALUATED        the injected expression WAS evaluated but the
                             oracle still didn't confirm (oracle gap to fix)
      NO_DIAG                no live diagnostic captured (older path)
    """
    diags = [a.get("diag") for a in attempts if a.get("diag")]
    if not diags:
        return "NO_DIAG"
    if any(d.get("proof_evaluated") for d in diags):
        return "PROOF_EVALUATED"
    if any(d.get("proof_conn_mismatch") for d in diags):
        return "PROOF_CONN_MISMATCH"
    if any(d.get("marker_in_executed_sql") for d in diags):
        # Split the old INPUT_IN_SQL_NO_PROOF by execution outcome: a
        # syntax/parse error on a marker-bearing statement means our payload
        # didn't fit the context (fixable); a clean run means the input was
        # accepted as data (likely a genuine negative). Mismatch takes
        # precedence -- it's the actionable signal.
        if any(d.get("marker_syntax_error") for d in diags):
            return "INPUT_IN_SQL_PAYLOAD_MISMATCH"
        if any(d.get("marker_ran_ok") for d in diags):
            return "INPUT_IN_SQL_RAN_SAFE"
        return "INPUT_IN_SQL_NO_PROOF"
    if any(d.get("n_sql_executed", 0) > 0 for d in diags):
        return "INPUT_NOT_IN_SQL"
    if any(d.get("function_entered") for d in diags):
        return "NO_SQL_EXECUTED"
    return "NOT_ENTERED"
