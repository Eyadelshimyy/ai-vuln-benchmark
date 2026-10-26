"""Opt-in local password vault: hygiene, not encryption."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jobbot.ops.redact import redact_text
from jobbot.vault import (
    VaultError,
    VaultPermissionError,
    can_type_password,
    delete_entry,
    init_vault,
    load_vault,
    needles_for_profile,
    normalize_site,
    put_entry,
    secret_needles,
    set_fill_login,
    vault_path,
)


def test_vault_sits_next_to_the_profile() -> None:
    assert vault_path(Path("data/profile.yaml")) == Path("data/.vault.yaml")
    assert vault_path(Path(".local/profile.yaml")) == Path(".local/.vault.yaml")


def test_init_creates_a_private_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "data" / ".vault.yaml"
    vault = init_vault(path)

    assert path.is_file()
    assert vault.sites() == ()
    assert vault.fill_login is False
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600


def test_init_refuses_to_clobber(tmp_path: Path) -> None:
    path = tmp_path / ".vault.yaml"
    init_vault(path)
    with pytest.raises(VaultError, match="already exists"):
        init_vault(path)


def test_group_readable_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / ".vault.yaml"
    init_vault(path)
    os.chmod(path, 0o644)
    with pytest.raises(VaultPermissionError, match="0600"):
        load_vault(path)


def test_symlink_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_text("version: 1\nfill_login: false\nentries: {}\n", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / ".vault.yaml"
    link.symlink_to(real)
    with pytest.raises(VaultPermissionError, match="symlink"):
        load_vault(link)


def test_put_never_invents_an_empty_password(tmp_path: Path) -> None:
    path = tmp_path / ".vault.yaml"
    init_vault(path)
    with pytest.raises(VaultError, match="never invents"):
        put_entry(path, "indeed", "")


def test_round_trip_hides_the_password_from_repr_and_list(tmp_path: Path) -> None:
    path = tmp_path / ".vault.yaml"
    init_vault(path)
    secret = "n0t-a-real-pass-w0rd"
    put_entry(path, "Indeed", secret, username="you@example.com")
    vault = load_vault(path)
    assert vault is not None
    assert vault.sites() == ("indeed",)
    entry = vault.get("indeed")
    assert entry is not None
    assert entry.reveal() == secret
    blob = f"{entry!r} {entry} {vault.sites()}"
    assert secret not in blob
    assert "password='***'" in repr(entry)


def test_fill_login_is_off_until_allowed(tmp_path: Path) -> None:
    path = tmp_path / ".vault.yaml"
    init_vault(path)
    put_entry(path, "indeed", "hunter2-plus")
    vault = load_vault(path)
    assert can_type_password(vault, "indeed") is False
    vault = set_fill_login(path, True)
    assert can_type_password(vault, "indeed") is True
    assert can_type_password(vault, "linkedin") is False
    vault = set_fill_login(path, False)
    assert can_type_password(vault, "indeed") is False


def test_delete_drops_only_that_site(tmp_path: Path) -> None:
    path = tmp_path / ".vault.yaml"
    init_vault(path)
    put_entry(path, "indeed", "aaa-secret")
    put_entry(path, "gmail", "bbb-secret")
    delete_entry(path, "indeed")
    vault = load_vault(path)
    assert vault is not None
    assert vault.sites() == ("gmail",)


def test_redact_strips_vault_values_from_failure_text(tmp_path: Path) -> None:
    profile = tmp_path / "data" / "profile.yaml"
    profile.parent.mkdir()
    profile.write_text("name: Test\n", encoding="utf-8")
    path = vault_path(profile)
    init_vault(path)
    secret = "unique-vault-token-xyz"
    put_entry(path, "indeed", secret)
    needles = needles_for_profile(profile)
    leaked = f"login failed password={secret} on indeed"
    out = redact_text(leaked, extra_secrets=needles)
    assert secret not in out
    assert "<redacted>" in out


def test_needles_are_longest_first() -> None:
    from jobbot.vault import Vault, VaultEntry

    vault = Vault(
        path=Path("x"),
        fill_login=False,
        entries={
            "a": VaultEntry("a", None, "abcd"),
            "b": VaultEntry("b", None, "abcdefgh"),
        },
    )
    assert secret_needles(vault) == ("abcdefgh", "abcd")


def test_site_keys_are_slugs() -> None:
    assert normalize_site("Get on Board") == "get-on-board"
    with pytest.raises(VaultError):
        normalize_site("")
    with pytest.raises(VaultError):
        normalize_site("***")


def test_missing_vault_is_not_an_error() -> None:
    assert load_vault(Path("/no/such/.vault.yaml"), missing_ok=True) is None
    assert can_type_password(None, "indeed") is False
    assert needles_for_profile(Path("/no/such/profile.yaml")) == ()
