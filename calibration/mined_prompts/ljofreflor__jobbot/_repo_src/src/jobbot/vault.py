"""Local opt-in credential vault — hygiene, not encryption (#157).

The OS user is the trust boundary. A hidden file is not a secret store: ``ls -a``
sees it, Drive would sync it, ``git add -f`` would publish it. JobBot never invents
a password, never prints one, never puts one on argv, and refuses to load a file
other users can read. CAPTCHA, 2FA, terms and submit stay human. Typing into a
login form is a second opt-in (``fill_login``) and is not implemented here.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VAULT_FILENAME = ".vault.yaml"
VAULT_VERSION = 1
MIN_SECRET_LEN = 4
_SITE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_HEADER = """\
# JobBot vault — local credentials. Never commit.
# Mode 0600. Hidden is not encryption: the OS user is the trust boundary.
# Do not sync this file to Drive/Dropbox. JobBot never invents these values.
# CAPTCHA, 2FA, terms and the final submit stay human even when fill_login is true.
"""


class VaultError(Exception):
    """The vault file is missing, unreadable, or not ours to trust."""


class VaultPermissionError(VaultError):
    """Group/other bits are set, or the path is not a regular file."""


class VaultMissingError(VaultError):
    """Opt-in file has not been created yet."""


@dataclass(frozen=True)
class VaultEntry:
    """One portal login. The password is never in ``repr`` / ``str``."""

    site: str
    username: str | None
    _password: str

    def reveal(self) -> str:
        """Password for a fill driver that already passed ``can_type_password``."""
        return self._password

    def __repr__(self) -> str:  # pragma: no cover - exercised via format
        user = self.username if self.username else "-"
        return f"VaultEntry(site={self.site!r}, username={user!r}, password='***')"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class Vault:
    path: Path
    fill_login: bool
    entries: dict[str, VaultEntry]

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def get(self, site: str) -> VaultEntry | None:
        return self.entries.get(normalize_site(site))


def vault_path(profile_path: Path) -> Path:
    """Sit next to ``profile.yaml`` so a workspace cannot read another's vault."""
    return profile_path.parent / VAULT_FILENAME


def normalize_site(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.strip().casefold()).strip("-.")
    if not slug or not _SITE_RE.fullmatch(slug):
        raise VaultError(f"invalid site key: {name!r}")
    return slug


def secret_needles(vault: Vault | None) -> tuple[str, ...]:
    """Password strings long enough to redact, longest first."""
    if vault is None:
        return ()
    found = {
        entry.reveal()
        for entry in vault.entries.values()
        if len(entry.reveal()) >= MIN_SECRET_LEN
    }
    return tuple(sorted(found, key=len, reverse=True))


def can_type_password(vault: Vault | None, site: str) -> bool:
    """True only with explicit fill_login *and* a stored entry. Drivers still HITL-submit."""
    if vault is None or not vault.fill_login:
        return False
    return vault.get(site) is not None


def _posix_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _assert_private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise VaultPermissionError(f"{path} must be a regular file (not a symlink)")
    if os.name != "posix":
        return
    mode = _posix_mode(path)
    if mode & 0o077:
        raise VaultPermissionError(
            f"{path} mode {mode:04o} allows group/other; chmod 0600 or recreate with secrets init"
        )


def _chmod_private(path: Path) -> None:
    if os.name == "posix":
        os.chmod(path, 0o600)


def _dump(vault: Vault) -> None:
    payload = {
        "version": VAULT_VERSION,
        "fill_login": vault.fill_login,
        "entries": {
            site: {
                **({"username": entry.username} if entry.username else {}),
                "password": entry.reveal(),
            }
            for site, entry in sorted(vault.entries.items())
        },
    }
    body = _HEADER + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    vault.path.parent.mkdir(parents=True, exist_ok=True)
    vault.path.write_text(body, encoding="utf-8")
    _chmod_private(vault.path)
    _assert_private_file(vault.path)


def _parse_entries(raw: Mapping[str, Any]) -> dict[str, VaultEntry]:
    block = raw.get("entries", {})
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise VaultError("entries must be a mapping of site → {password, username?}")
    entries: dict[str, VaultEntry] = {}
    for key, value in block.items():
        site = normalize_site(str(key))
        if not isinstance(value, dict):
            raise VaultError(f"entry {site!r} must be a mapping")
        password = value.get("password")
        if not isinstance(password, str) or not password:
            raise VaultError(f"entry {site!r} has no password (JobBot never invents one)")
        username_raw = value.get("username")
        username = (
            username_raw.strip()
            if isinstance(username_raw, str) and username_raw.strip()
            else None
        )
        entries[site] = VaultEntry(site=site, username=username, _password=password)
    return entries


def load_vault(path: Path, *, missing_ok: bool = False) -> Vault | None:
    """Read the vault. ``missing_ok`` returns None when the file does not exist."""
    if not path.exists():
        if missing_ok:
            return None
        raise VaultMissingError(f"no vault at {path} — run jobbot secrets init")
    _assert_private_file(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise VaultError(f"{path} is not a YAML mapping")
    version = loaded.get("version", VAULT_VERSION)
    if version != VAULT_VERSION:
        raise VaultError(f"unsupported vault version: {version!r}")
    return Vault(
        path=path,
        fill_login=loaded.get("fill_login") is True,
        entries=_parse_entries(loaded),
    )


def require_vault(path: Path) -> Vault:
    vault = load_vault(path, missing_ok=False)
    if vault is None:
        raise VaultMissingError(f"no vault at {path} — run jobbot secrets init")
    return vault


def init_vault(path: Path, *, exist_ok: bool = False) -> Vault:
    """Create an empty private vault. Does not invent entries."""
    if path.exists() and not exist_ok:
        raise VaultError(f"vault already exists: {path}")
    if path.exists():
        return require_vault(path)
    vault = Vault(path=path, fill_login=False, entries={})
    _dump(vault)
    return vault


def put_entry(
    path: Path,
    site: str,
    password: str,
    *,
    username: str | None = None,
) -> Vault:
    """Store a human-supplied password. Empty values are refused, never invented."""
    if not password:
        raise VaultError("password is empty; JobBot never invents one")
    vault = require_vault(path)
    key = normalize_site(site)
    user = username.strip() if username and username.strip() else None
    entries = dict(vault.entries)
    entries[key] = VaultEntry(site=key, username=user, _password=password)
    updated = Vault(path=vault.path, fill_login=vault.fill_login, entries=entries)
    _dump(updated)
    return updated


def delete_entry(path: Path, site: str) -> Vault:
    vault = require_vault(path)
    key = normalize_site(site)
    if key not in vault.entries:
        raise VaultError(f"no entry for {key}")
    entries = dict(vault.entries)
    del entries[key]
    updated = Vault(path=vault.path, fill_login=vault.fill_login, entries=entries)
    _dump(updated)
    return updated


def set_fill_login(path: Path, enabled: bool) -> Vault:
    vault = require_vault(path)
    updated = Vault(path=vault.path, fill_login=enabled, entries=dict(vault.entries))
    _dump(updated)
    return updated


def needles_for_profile(profile_path: Path) -> tuple[str, ...]:
    """Best-effort needles for ops redaction; never raises."""
    try:
        return secret_needles(load_vault(vault_path(profile_path), missing_ok=True))
    except VaultError:
        return ()
    except OSError:
        return ()


def redact_known_secrets(text: str, secrets: Iterable[str]) -> str:
    """Replace stored passwords that appear as raw substrings."""
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "<redacted>")
    return out
