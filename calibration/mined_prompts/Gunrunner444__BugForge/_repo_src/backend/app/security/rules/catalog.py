"""Built-in security rule catalog."""

from __future__ import annotations

from app.security.rules.base import SecurityRule
from app.security.rules.controls import (
    CsrfDisabledRule,
    JwtUnverifiedRule,
    PasswordCompareRule,
    UnscopedLookupRule,
)
from app.security.rules.indicators import HardcodedSecretRule, InsecureConfigRule, WeakCryptoRule
from app.security.rules.taint_rules import default_taint_rules
from app.security.solidity.rules import solidity_security_rules


def builtin_security_rules() -> list[SecurityRule]:
    return [
        *default_taint_rules(),
        *solidity_security_rules(),
        HardcodedSecretRule(),
        WeakCryptoRule(),
        InsecureConfigRule(),
        JwtUnverifiedRule(),
        PasswordCompareRule(),
        CsrfDisabledRule(),
        UnscopedLookupRule(),
    ]
