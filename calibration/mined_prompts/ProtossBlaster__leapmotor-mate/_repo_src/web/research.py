"""Research / BetaTester mode — full-signal capture + encrypted export for REEV support.

This module is INERT unless MATE_RESEARCH is on (only the MateBetaTesterOnly image sets it).
The official build ships this code but never activates it, so normal users get nothing.

Why it exists: the BEV pipeline mis-handles REEV behaviour (e.g. the range-extender charging
the battery while driving looks like a negative-efficiency trip), so we cannot integrate REEV
into trips/charges/stats by guessing. The beta build captures every raw signal over time, paired
with the tester's logbook + official-app screenshots, so we can map the REEV signals for real and
then ship proper REEV support in the normal Mate.

Export crypto (asymmetric, so the build holds no secret): a random Fernet key encrypts the bundle,
then the bundle's Fernet key is sealed to our RSA public key (shipped in research/). Only the
matching PRIVATE key — kept off-repo on the maintainer's Mac — can open an exported bundle. So even
if a bundle leaks, nobody but us can read it. Uses `cryptography` (already a dependency).
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path

# GPS signal IDs (signed + unsigned lat/lon) — stripped from every export so a tester's
# location history never leaves their machine. The fuel/range signals we actually need carry
# no location. VIN/account live in other tables and are never part of the signal log.
REDACT_SIGNALS = frozenset({"2", "3", "3724", "3725", "2190", "2191"})

_PUBLIC_KEY = Path(__file__).resolve().parent / "research" / "beta_public_key.pem"


def research_enabled() -> bool:
    """True only in the MateBetaTesterOnly build (env baked into that image)."""
    return os.environ.get("MATE_RESEARCH", "") not in ("", "0", "false", "False", "no")


def redact_signal_rows(rows):
    """Drop location signals from exported (ts, key, value) rows."""
    return [r for r in rows if (r[1] if not isinstance(r, dict) else r.get("key")) not in REDACT_SIGNALS]


def encrypt_bundle(plaintext: bytes) -> bytes:
    """Seal `plaintext` so only the maintainer's private key can open it. Returns a small
    JSON envelope (RSA-OAEP-sealed Fernet key + Fernet-encrypted data), safe to attach to a
    public issue."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    pub = serialization.load_pem_public_key(_PUBLIC_KEY.read_bytes())
    fkey = Fernet.generate_key()
    token = Fernet(fkey).encrypt(plaintext)
    sealed_key = pub.encrypt(
        fkey,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None),
    )
    return json.dumps({
        "v": 1,
        "alg": "RSA-OAEP-SHA256+Fernet",
        "key": base64.b64encode(sealed_key).decode(),
        "data": base64.b64encode(token).decode(),
    }).encode()
