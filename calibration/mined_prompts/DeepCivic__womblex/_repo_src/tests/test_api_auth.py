"""Service-token auth: registry validation, token matching, scopes, the CLI verb."""
from __future__ import annotations

import pytest
import yaml

from womblex.api.auth import (
    Caller,
    ClientRegistry,
    RegistryError,
    generate_token,
    hash_token,
    load_registry,
    parse_registry,
    registry_from_env,
)
from womblex.cli import main


def _entry(client_id="redline", token="tok", scopes=("submit", "read")):
    return {"client_id": client_id, "token_sha256": hash_token(token), "scopes": list(scopes)}


def test_generated_token_hashes_to_its_digest():
    token, digest = generate_token()
    assert token.startswith("womblex_") and digest == hash_token(token)
    assert generate_token()[0] != token


def test_authenticate_matches_only_the_right_token():
    registry = parse_registry({"clients": [_entry("a", "ta"), _entry("b", "tb", ["admin"])]})
    assert registry.authenticate("ta") == Caller("a", frozenset({"submit", "read"}))
    assert registry.authenticate("tb").client_id == "b"
    assert registry.authenticate("nope") is None
    assert registry.authenticate("") is None


def test_admin_implies_every_scope_and_reads_unscoped():
    admin = Caller("ops", frozenset({"admin"}))
    assert admin.has("read_raw") and admin.owner is None
    caller = Caller("redline", frozenset({"read"}))
    assert caller.has("read") and not caller.has("read_raw")
    assert caller.owner == "redline"


@pytest.mark.parametrize("doc, message", [
    (None, "clients"),
    ({"clients": "x"}, "clients"),
    ({"clients": [{"client_id": "", "token_sha256": "0" * 64, "scopes": ["read"]}]}, "client_id"),
    ({"clients": [{**_entry(), "token_sha256": "ABC"}]}, "64 lowercase hex"),
    ({"clients": [{**_entry(), "scopes": []}]}, "scopes"),
    ({"clients": [{**_entry(), "scopes": ["root"]}]}, "unknown scope"),
    ({"clients": [_entry(), _entry()]}, "duplicate"),
])
def test_malformed_registries_are_refused(doc, message):
    with pytest.raises(RegistryError, match=message):
        parse_registry(doc)


def test_load_registry_and_env(tmp_path, monkeypatch):
    path = tmp_path / "clients.yaml"
    path.write_text(yaml.safe_dump({"clients": [_entry()]}))
    assert load_registry(path).authenticate("tok").client_id == "redline"
    monkeypatch.setenv("WOMBLEX_API_CLIENTS", str(path))
    assert not registry_from_env().empty
    monkeypatch.delenv("WOMBLEX_API_CLIENTS")
    assert registry_from_env().empty
    with pytest.raises(RegistryError, match="cannot read"):
        load_registry(tmp_path / "missing.yaml")


def test_api_token_verb_prints_a_token_whose_hash_is_in_the_entry(capsys):
    assert main(["api-token", "--client", "redline", "--scope", "read"]) == 0
    out = capsys.readouterr().out
    token = out.split("shown once; give it to the client): ")[1].splitlines()[0]
    assert f"token_sha256: {hash_token(token)}" in out
    assert "client_id: redline" in out and "scopes: [read]" in out
    entry = yaml.safe_load("clients:\n" + out.split("`clients:`):\n")[1])
    assert parse_registry(entry).authenticate(token).client_id == "redline"


def test_api_token_verb_defaults_scopes_and_rejects_a_blank_client(capsys):
    assert main(["api-token", "--client", "x"]) == 0
    assert "scopes: [submit, read]" in capsys.readouterr().out
    assert main(["api-token", "--client", " "]) == 1


# --- FastAPI dependencies ------------------------------------------------------

fastapi = pytest.importorskip("fastapi")


def _app(registry):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from womblex.api.auth import caller_dependency, require_scope

    get_caller = caller_dependency(registry)
    app = FastAPI()

    @app.get("/who")
    def who(caller: Caller = Depends(get_caller)):  # noqa: B008
        return {"client_id": caller.client_id}

    @app.get("/raw")
    def raw(caller: Caller = Depends(require_scope(get_caller, "read_raw"))):  # noqa: B008
        return {"client_id": caller.client_id}

    return TestClient(app)


def _registry():
    return ClientRegistry(parse_registry({"clients": [
        _entry("reader", "t-read", ["read"]),
        _entry("rawer", "t-raw", ["read_raw"]),
    ]}).clients)


def test_missing_or_wrong_token_is_401():
    client = _app(_registry())
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic t-read"},
                    {"Authorization": "Bearer"}):
        resp = client.get("/who", headers=headers)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"


def test_valid_token_resolves_the_caller():
    resp = _app(_registry()).get("/who", headers={"Authorization": "bearer t-read"})
    assert resp.status_code == 200 and resp.json() == {"client_id": "reader"}


def test_a_missing_scope_is_403_and_a_held_scope_passes():
    client = _app(_registry())
    assert client.get("/raw", headers={"Authorization": "Bearer t-read"}).status_code == 403
    assert client.get("/raw", headers={"Authorization": "Bearer t-raw"}).status_code == 200
    assert client.get("/raw").status_code == 401


def test_insecure_mode_admits_every_request_as_an_all_scope_caller():
    client = _app(None)
    assert client.get("/who").json() == {"client_id": "insecure"}
    assert client.get("/raw").status_code == 200
