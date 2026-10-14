"""Tests for Isaacus client construction — hosted API vs SageMaker.

Endpoint-spec parsing, model-reachability and the availability gate are pure
functions of the environment, so they run everywhere with no SDK, no AWS
credentials and no network. Building an actual SageMaker-backed client needs
``isaacus`` + ``isaacus-sagemaker`` installed and is skipped without them.
"""

from __future__ import annotations

import pytest

from womblex.utils.availability import isaacus_available
from womblex.utils.isaacus_client import (
    ENDPOINTS_ENV,
    PROFILE_ENV,
    REGION_ENV,
    SageMakerEndpoint,
    endpoints_from_env,
    is_model_served,
    make_ai_chunking_client,
    parse_endpoints,
    sagemaker_configured,
    unserved_models,
)

# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parse_universal_endpoint_serves_all_models():
    """No `=models` part is the universal subscription: every model routes there."""
    (endpoint,) = parse_endpoints("kanon-2-universal-001")
    assert endpoint == SageMakerEndpoint(name="kanon-2-universal-001")
    assert endpoint.models is None
    assert is_model_served("kanon-2-embedder", [endpoint])
    assert is_model_served("anything-at-all", [endpoint])


def test_parse_per_feature_endpoints():
    embed, enrich = parse_endpoints(
        "embed-001=kanon-2-embedder, enrich-001=kanon-2-enricher|kanon-2-classifier"
    )
    assert embed == SageMakerEndpoint(name="embed-001", models=("kanon-2-embedder",))
    assert enrich.models == ("kanon-2-enricher", "kanon-2-classifier")
    assert is_model_served("kanon-2-classifier", [embed, enrich])
    assert not is_model_served("kanon-2-reranker", [embed, enrich])


def test_parse_per_endpoint_region():
    (endpoint,) = parse_endpoints("embed-001@ap-southeast-2=kanon-2-embedder")
    assert endpoint.name == "embed-001"
    assert endpoint.region == "ap-southeast-2"
    assert endpoint.models == ("kanon-2-embedder",)


def test_parse_ignores_blank_entries_and_whitespace():
    assert parse_endpoints("  ,, a-001 , ") == [SageMakerEndpoint(name="a-001")]
    assert parse_endpoints("") == []


@pytest.mark.parametrize("spec", ["=kanon-2-embedder", "embed-001="])
def test_parse_rejects_malformed_specs(spec):
    """Malformed is loud: silently ignoring it would fall back to a key that an
    air-gapped deployment does not have."""
    with pytest.raises(ValueError):
        parse_endpoints(spec)


# ---------------------------------------------------------------------------
# Environment-driven behaviour
# ---------------------------------------------------------------------------


def test_sagemaker_not_configured_by_default(monkeypatch):
    monkeypatch.delenv(ENDPOINTS_ENV, raising=False)
    assert endpoints_from_env() == []
    assert sagemaker_configured() is False
    # Every model is reachable on the hosted API, so nothing is reported unserved.
    assert unserved_models(["kanon-2-embedder"]) == []


def test_unserved_models_reports_undeployed_subscriptions(monkeypatch):
    monkeypatch.setenv(ENDPOINTS_ENV, "embed-001=kanon-2-embedder")
    assert sagemaker_configured() is True
    assert unserved_models(["kanon-2-embedder"]) == []
    assert unserved_models(["kanon-2-enricher", "kanon-2-embedder"]) == ["kanon-2-enricher"]


def test_availability_gate_needs_no_key_on_sagemaker(monkeypatch):
    """SageMaker deployments are air-gapped — there is no API key to check."""
    pytest.importorskip("isaacus")
    monkeypatch.delenv("ISAACUS_API_KEY", raising=False)
    monkeypatch.delenv(ENDPOINTS_ENV, raising=False)
    assert isaacus_available() is False
    monkeypatch.setenv(ENDPOINTS_ENV, "kanon-2-universal-001")
    assert isaacus_available() is True


def test_ai_chunking_client_is_none_without_chunking_model():
    assert make_ai_chunking_client(None) is None
    assert make_ai_chunking_client("") is None


# ---------------------------------------------------------------------------
# Client construction (needs the SDK + integration package)
# ---------------------------------------------------------------------------


def test_hosted_api_client_strips_a_pasted_key(monkeypatch):
    """The whitespace strip moved modules with the factory — a trailing newline
    otherwise reaches httpx as an illegal header value."""
    pytest.importorskip("isaacus")
    from womblex.utils.isaacus_client import make_isaacus_client

    monkeypatch.delenv(ENDPOINTS_ENV, raising=False)
    # Deliberately not key-shaped. The strip is what is under test, and a value
    # wearing the real `iuak_` prefix reads as a live credential to every scanner
    # that walks this repo.
    monkeypatch.setenv("ISAACUS_API_KEY", "trailing-whitespace-probe\n")
    client = make_isaacus_client(models=["kanon-2-embedder"])
    assert client.api_key == "trailing-whitespace-probe"  # pragma: allowlist secret


def test_sagemaker_client_rejects_undeployed_model(monkeypatch):
    pytest.importorskip("isaacus")
    pytest.importorskip("isaacus_sagemaker")
    from womblex.utils.isaacus_client import make_isaacus_client

    monkeypatch.setenv(ENDPOINTS_ENV, "embed-001=kanon-2-embedder")
    monkeypatch.setenv(REGION_ENV, "ap-southeast-2")
    with pytest.raises(ValueError, match="kanon-2-enricher"):
        make_isaacus_client(models=["kanon-2-enricher"])


def test_sagemaker_client_builds_without_api_key(monkeypatch):
    """The integration signs with AWS credentials; no Isaacus key is involved."""
    pytest.importorskip("isaacus")
    pytest.importorskip("isaacus_sagemaker")
    from womblex.utils.isaacus_client import make_isaacus_client

    monkeypatch.delenv("ISAACUS_API_KEY", raising=False)
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    monkeypatch.setenv(ENDPOINTS_ENV, "embed-001=kanon-2-embedder,universal-001")
    monkeypatch.setenv(REGION_ENV, "ap-southeast-2")
    client = make_isaacus_client(models=["kanon-2-embedder", "kanon-2-enricher"])
    # The enricher resolves through the catch-all endpoint, the embedder through
    # its own — both without a key, and without any request being made.
    assert client.base_url is not None


def test_sagemaker_client_requires_a_resolvable_region(monkeypatch):
    """A missing region otherwise surfaces as a bare AssertionError per request."""
    pytest.importorskip("isaacus")
    pytest.importorskip("isaacus_sagemaker")
    from womblex.utils.isaacus_client import make_isaacus_client

    monkeypatch.setenv(ENDPOINTS_ENV, "universal-001")
    monkeypatch.delenv(REGION_ENV, raising=False)
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    monkeypatch.setattr("boto3.Session", lambda **_kw: _NoRegionSession())
    with pytest.raises(ValueError, match="No AWS region"):
        make_isaacus_client()


class _NoRegionSession:
    """Stands in for an AWS session with no region configured anywhere."""

    region_name = None
