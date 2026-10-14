"""MQTT 'Test connection' reuses the SAVED password when the masked field is empty — GitHub #91.

The password input renders as •••••••• and never carries the real value, so clicking Test
without retyping submits an empty mqtt_pass. Before the fix the endpoint tested with NO
password → the broker replied 'Not authorised' (red), while the running bridge AND the
status dot — both on the saved secret — stayed green. That is the exact mismatch riri19
reported (#91). The fix mirrors Save and the status dot: empty/absent field → saved
password; a typed value is still used verbatim (so testing new creds before saving works).

Needs web.main (fastapi); the minimal CI env skips this module cleanly.
"""
import asyncio
import pytest

pytest.importorskip("fastapi", reason="web.main needs fastapi (absent in the minimal CI test env)")

import db as D
import db_reader
import main
import mqtt_check


class _Req:
    """Minimal stand-in for a Starlette Request — test_mqtt only awaits .form()."""
    def __init__(self, data):
        self._data = data

    async def form(self):
        return self._data


def _password_used(tmp_path, monkeypatch, form, saved=None):
    """Wire db_reader to a fresh DB, optionally store a saved mqtt_pass, then invoke the
    real test_mqtt endpoint and return the password it handed to check_connection."""
    D.Database(str(tmp_path / "t.db"))
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    if saved is not None:
        db_reader.set_secret("mqtt_pass", saved)
    seen = {}

    def fake_check(broker, port, user, password, tls, tls_insecure):
        seen["password"] = password
        return True, ""

    monkeypatch.setattr(mqtt_check, "check_connection", fake_check)
    asyncio.run(main.test_mqtt(_Req(form)))
    return seen["password"]


_FORM = {"mqtt_broker": "10.0.0.1", "mqtt_port": "1883", "mqtt_user": "u"}


def test_empty_field_falls_back_to_saved_password(tmp_path, monkeypatch):
    pw = _password_used(tmp_path, monkeypatch, {**_FORM, "mqtt_pass": ""}, saved="saved-secret")
    assert pw == "saved-secret"          # the #91 fix: not None


def test_absent_field_falls_back_to_saved_password(tmp_path, monkeypatch):
    # Some browsers drop an empty masked field from the submit entirely — same outcome.
    pw = _password_used(tmp_path, monkeypatch, dict(_FORM), saved="saved-secret")
    assert pw == "saved-secret"


def test_whitespace_only_field_falls_back_to_saved_password(tmp_path, monkeypatch):
    pw = _password_used(tmp_path, monkeypatch, {**_FORM, "mqtt_pass": "   "}, saved="saved-secret")
    assert pw == "saved-secret"


def test_typed_password_is_used_verbatim(tmp_path, monkeypatch):
    # Testing NEW credentials before saving must still use exactly what was typed.
    pw = _password_used(tmp_path, monkeypatch, {**_FORM, "mqtt_pass": "typed-pw"}, saved="saved-secret")
    assert pw == "typed-pw"


def test_no_password_anywhere_stays_none(tmp_path, monkeypatch):
    # No saved secret and an empty field → genuinely password-less broker (None).
    pw = _password_used(tmp_path, monkeypatch, {**_FORM, "mqtt_pass": ""}, saved=None)
    assert pw is None
