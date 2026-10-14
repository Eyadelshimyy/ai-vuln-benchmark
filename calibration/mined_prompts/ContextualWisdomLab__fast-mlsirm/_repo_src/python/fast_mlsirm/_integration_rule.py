"""Bounded normalization for public item-model integration-rule controls."""

from __future__ import annotations


_NODE_RULE_ALIASES = {
    "gh": "gh",
    "gauss-hermite": "gh",
    "gausshermite": "gh",
    "qmc": "qmc",
    "mc": "mc",
}
_NODE_RULE_ERROR = "node_rule must be one of the supported integration rules"


def normalize_node_rule(node_rule: str) -> str:
    """Return the canonical integration-rule name without reflecting caller data.

    Only the documented Gauss-Hermite aliases plus ``qmc`` and ``mc`` are
    accepted.  Non-string or unsupported controls fail with a package-owned
    message; rejected values are never coerced through ``str``/``repr``.
    ``str.lower`` is called as the built-in descriptor so a string subclass
    cannot replace normalization with a caller-defined ``lower`` hook.
    """
    if type(node_rule) is not str:
        raise ValueError(_NODE_RULE_ERROR)
    canonical = _NODE_RULE_ALIASES.get(str.lower(node_rule))
    if canonical is None:
        raise ValueError(_NODE_RULE_ERROR)
    return canonical
