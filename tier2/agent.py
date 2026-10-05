"""Optional agentic payload proposer for Tier-2.

The agent CRAFTS a candidate exploit payload; it never decides the verdict --
the deterministic observable check in the engine does. This keeps the project's
invariant intact: detection/confirmation is never agentic, only exploit
construction is. Off by default (the deterministic bank runs first and alone
unless --use-agent is passed), so the whole pipeline is reproducible offline.

The caller (`call`) is generate_batch.make_caller(base_url, model): a function
call(messages, temperature) -> str. Any OpenAI-compatible model works (local
Ollama or a hosted one).
"""
from __future__ import annotations

import re


def _clean(raw: str) -> str:
    """Pull a single payload value out of a model reply: drop markdown fences,
    surrounding quotes, and leading 'payload:' style labels."""
    s = raw.strip()
    m = re.search(r"```(?:\w+)?\n?(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    # take the first non-empty line if the model rambled
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if lines:
        s = lines[0].strip()
    s = re.sub(r"^(payload|value|answer)\s*[:=]\s*", "", s, flags=re.IGNORECASE).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return s


def propose_payload(call, record: dict, param: str, strat, ctx, feedback: str | None) -> str:
    """Ask the model for one attacker-controlled value for `param` that should
    achieve the strategy's (concrete, rendered) observable goal."""
    source = record.get("generated_source") or record.get("ground_truth_source") or ""
    goal = ctx.render(strat.goal)
    system = (
        "You are a security researcher writing a proof-of-concept exploit in a "
        "sandboxed lab. You are given one Python function and the name of one "
        "parameter that is fully attacker-controlled. Produce a single concrete "
        "value for that parameter that achieves the stated goal when the function "
        "runs. Respond with ONLY the raw value -- no code fences, no quotes, no "
        "explanation."
    )
    user = (
        f"Function:\n\n{source}\n\n"
        f"Attacker-controlled parameter: {param!r}\n"
        f"Vulnerability class: {strat.cwe} ({strat.name})\n"
        f"Goal: {goal}\n"
    )
    if feedback:
        user += f"\nYour previous attempt did not work: {feedback}\nTry a different break-out sequence."
    user += "\n\nOutput only the single parameter value."
    raw = call([{"role": "system", "content": system}, {"role": "user", "content": user}], 0.4)
    return _clean(raw)
