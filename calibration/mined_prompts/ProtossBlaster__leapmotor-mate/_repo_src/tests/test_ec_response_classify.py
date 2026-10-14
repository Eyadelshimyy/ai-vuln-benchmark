"""getEC response classifier — the fix for the Convert-button / sweep false negative.

A long-lived web session's cloud token expires; getLastweekEC then returns a non-data response that
the old code misread as "no data" → it reported "no official data" for trips that DO have it. The
classifier separates a real split, a genuine empty window, and an auth/token failure (→ refresh+retry).
Verified response shapes: success {result:0,code:0,data:{...}}; no-data {result:100,code:100,
message:'No data found'}.
"""
import command_client as cc


def test_real_data_split():
    j = {"result": 0, "code": 0, "data": {"driverEC": "0.8", "acEC": "0.9", "otherEC": "0.2"}}
    kind, d = cc._classify_ec_response(j)
    assert kind == "data" and d["driverEC"] == "0.8"


def test_genuine_no_data_message():
    kind, d = cc._classify_ec_response({"result": 100, "code": 100, "message": "No data found"})
    assert kind == "empty" and d is None


def test_success_code_but_empty_window():
    kind, _ = cc._classify_ec_response({"result": 0, "code": 0})
    assert kind == "empty"


def test_expired_token_is_auth_not_empty():
    """The crux: a token error must NOT be read as 'no data' (would block the retry/refresh)."""
    kind, _ = cc._classify_ec_response({"result": 401, "code": 401, "message": "token invalid"})
    assert kind == "auth"


def test_garbage_or_missing_body_is_auth():
    assert cc._classify_ec_response(None)[0] == "auth"
    assert cc._classify_ec_response({})[0] == "auth"
