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

    The buckets are grouped by CWE family; the SQL ones are kept exactly as
    before, and path (CWE-22) / command (CWE-78) ones were added so those
    classes stop collapsing to the misleading SQL-only "no SQL executed".

      NOT_ENTERED            the candidate function never ran in live mode
                             (construction / import / CLI-interface failure)
      EXECUTED_NO_SINK       it ran to completion but touched no tracked
                             injection sink at all (SQL, filesystem, or process)
                             -- for a flagged function this points at a likely
                             LLM false positive or an un-instrumented sink
      SINK_REACHED_INPUT_ABSENT  a filesystem/process sink ran, but our payload
                             was not in its argument -> taint did not propagate
                             from the parameter to the sink (flow broke / the
                             value was sanitised or replaced before the sink)
      PATH_TRAVERSAL_NO_OBSERVABLE  our payload reached open()/Path.write_* AND
                             the resolved path escaped the sandbox cwd, yet the
                             observable did not fire -> almost certainly a real
                             traversal the oracle missed -> ORACLE GAP, FIX
      PATH_SINK_INPUT_BOUNDED  payload reached a filesystem sink but the path
                             stayed inside the sandbox -> input accepted but
                             neutralised -> LIKELY A GENUINE NEGATIVE
      CMD_SINK_REACHED_NO_OBSERVABLE  payload reached os.system/os.popen/Popen
                             but no observable effect fired -> oracle/payload gap
                             for this command context
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
      LIVE_RUN_FAILED        no execution-chain diag was captured because the
                             candidate could not be run in live mode at all
                             (timeout, child crash, construction/interface gap)
                             -- a harness/interface gap to fix, never a pass
      NO_DIAG                no diag and no verdict to explain it (should be rare)
    """
    diags = [a.get("diag") for a in attempts if a.get("diag")]
    if not diags:
        # No execution-chain diag means nothing ran in live mode. A live
        # NOT_TRIGGERED / TRIGGERED always attaches a diag, so the only way here
        # is a COULD_NOT_EXECUTE (timeout, child crash, un-runnable interface).
        # Surface that as a real root cause instead of the old blank NO_DIAG --
        # it is a harness gap to close, not a silent pass, and crucially it is
        # still NOT a confirmation: nothing here is counted exploited.
        if any(a.get("tier1_verdict") == "COULD_NOT_EXECUTE" for a in attempts):
            return "LIVE_RUN_FAILED"
        if any(a.get("source") == "agent" and a.get("error") for a in attempts):
            return "LIVE_RUN_FAILED"
        return "NO_DIAG"
    # --- SQL (CWE-89) chain, strongest signal first ---
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
    # --- path traversal (CWE-22): did our payload reach a filesystem sink? ---
    if any(d.get("marker_in_path_sink") for d in diags):
        # Payload reached open()/Path.write_*. If the resolved path escaped the
        # sandbox cwd, this is almost certainly a real traversal the observable
        # just didn't capture -> an ORACLE GAP to fix. If it stayed in-bounds,
        # the input was accepted but neutralised -> likely a genuine negative.
        if any(d.get("path_escaped") for d in diags):
            return "PATH_TRAVERSAL_NO_OBSERVABLE"
        return "PATH_SINK_INPUT_BOUNDED"
    # --- command injection (CWE-78): did our payload reach a process sink? ---
    if any(d.get("marker_in_cmd_sink") for d in diags):
        # A shell context (os.system/os.popen/Popen shell=True) ran our input
        # but the injected command left no effect -> the breakout did not fire
        # -> a payload/oracle gap worth a better breakout (FIXABLE). Reaching a
        # NON-shell Popen means our input was an argv element, never parsed as a
        # command -> not an injection for this call shape -> a genuine negative.
        if any(d.get("cmd_shell_reached") for d in diags):
            return "CMD_SHELL_NO_EFFECT"
        return "CMD_ARGV_NOT_SHELL"
    if any(d.get("n_sql_executed", 0) > 0 for d in diags):
        return "INPUT_NOT_IN_SQL"
    # A dangerous sink ran but our payload was not in it -> taint did not
    # propagate from the parameter to the sink (broke flow / sanitised early).
    if any(d.get("n_path_sinks", 0) > 0 or d.get("n_cmd_sinks", 0) > 0 for d in diags):
        return "SINK_REACHED_INPUT_ABSENT"
    if any(d.get("function_entered") for d in diags):
        # Ran to completion but touched no tracked injection sink at all.
        return "EXECUTED_NO_SINK"
    return "NOT_ENTERED"
