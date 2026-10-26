"""Tests for `/api/dataset-groups` create + list."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.models.enums import CombinationMethod, ErrorCode, WarningCode


client = TestClient(app)


def test_create_multi_peril_max_attaches_max_across_perils_warning() -> None:
    payload = {
        "groupName": "Farmers 2027 All Perils",
        "currency": "USD",
        "combinationMethod": CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN.value,
        "members": [
            {"datasetId": "ds-farmers-bda-2027", "peril": "WS"},
            {"datasetId": "ds-farmers-bda-2027", "peril": "EQ"},
            {"datasetId": "ds-acmere-26-multi", "peril": "CS"},
        ],
    }
    resp = client.post("/api/dataset-groups", json=payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["datasetGroupId"].startswith("grp-")
    codes = {w["code"] for w in body["warnings"]}
    assert WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS.value in codes


def test_create_mixed_currency_returns_409_currency_mismatch() -> None:
    payload = {
        "groupName": "Mixed Currency",
        "currency": "USD",
        "combinationMethod": CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN.value,
        "members": [
            {"datasetId": "ds-farmers-bda-2027", "peril": "WS"},
            {"datasetId": "ds-sample-27-ws", "peril": "WS"},
        ],
    }
    resp = client.post("/api/dataset-groups", json=payload)
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["error"]["code"] == ErrorCode.CURRENCY_MISMATCH.value


def test_create_sum_distinct_without_confirmation_returns_422() -> None:
    payload = {
        "groupName": "Sum Distinct (unconfirmed)",
        "currency": "USD",
        "combinationMethod": CombinationMethod.SUM_DISTINCT_SEGMENTS.value,
        "members": [
            {"datasetId": "ds-farmers-bda-2027", "peril": "WS"},
            {"datasetId": "ds-farmers-bda-2027", "peril": "EQ"},
        ],
    }
    resp = client.post("/api/dataset-groups", json=payload)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == ErrorCode.VALIDATION_ERROR.value


def test_create_then_list_includes_new_group() -> None:
    payload = {
        "groupName": "List-Test Group",
        "currency": "USD",
        "combinationMethod": CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN.value,
        "members": [
            {"datasetId": "ds-farmers-bda-2027", "peril": "WS"},
            {"datasetId": "ds-farmers-bda-2027", "peril": "EQ"},
        ],
    }
    created = client.post("/api/dataset-groups", json=payload)
    assert created.status_code == 201
    new_id = created.json()["datasetGroupId"]

    listed = client.get("/api/dataset-groups")
    assert listed.status_code == 200
    ids = {g["datasetGroupId"] for g in listed.json()["datasetGroups"]}
    assert new_id in ids
