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
                             "tier1_verdict": res.get("verdict"), "proved": proved})
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

    return {"confirmed": False, "cwe": cwe, "param": param, "attempts": attempts}
