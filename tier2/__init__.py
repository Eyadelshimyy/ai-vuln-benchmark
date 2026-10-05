"""VULCAN Tier-2: execution-PROVEN exploitation.

Tier 1 (the deterministic oracle) says "a tainted value reached a dangerous
sink under an attacker-controllability assumption." Tier 2 goes further and
PROVES it: it runs a real attacker payload through the candidate's real code
in live mode and checks for a real, observable malicious effect (a file the
attacker's command created, a table an injected query created, a write that
escaped its intended directory). Only when that effect is observed does Tier 2
report CONFIRMED_EXPLOITABLE.

Design rule (non-negotiable): the DETECTION oracle is never agentic. Tier 2
may use an LLM to *craft* the exploit payload, but a deterministic observable
check -- not the model's say-so, and not the sink merely firing -- decides
whether the exploit is confirmed. "Prove it's vulnerable, don't just state it."
"""
