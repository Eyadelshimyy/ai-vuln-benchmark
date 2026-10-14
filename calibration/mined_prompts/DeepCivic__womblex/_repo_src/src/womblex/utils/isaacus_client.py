"""Isaacus client construction — hosted API or private SageMaker deployment.

One factory, :func:`make_isaacus_client`, covers both deployments:

- **Hosted API** (default): the SDK reads ``ISAACUS_API_KEY``.
- **SageMaker** (air-gapped, inside the customer's own AWS account): set
  ``ISAACUS_SAGEMAKER_ENDPOINTS`` and the SDK is bootstrapped with
  ``isaacus_sagemaker.IsaacusSageMakerRuntimeHTTPClient``, which proxies every
  request through the endpoint's ``/invocations`` path. No API key is involved
  — that HTTP client stamps a placeholder over ``ISAACUS_API_KEY`` when it is
  unset, so the SDK's own key check passes.

Marketplace subscriptions are per model (``kanon-2-embedder``,
``kanon-2-enricher``, …) plus a universal one, so a deployment ranges from one
endpoint serving everything to one per feature, mixed. Womblex assumes no
shape: the env var declares comma-separated
``name[@region][=model|model|...]`` entries, an entry with no ``=models`` part
serving every model (the integration's own default)::

    ISAACUS_SAGEMAKER_ENDPOINTS=kanon-2-universal-001
    ISAACUS_SAGEMAKER_ENDPOINTS=embed-001=kanon-2-embedder,enrich-001=kanon-2-enricher
    ISAACUS_SAGEMAKER_ENDPOINTS=embed-001@ap-southeast-2=kanon-2-embedder

Region falls back to ``ISAACUS_SAGEMAKER_REGION`` then the AWS SDK's own
resolution; ``ISAACUS_SAGEMAKER_PROFILE`` selects a non-default profile.
Callers pass the model ids they will call, so an undeployed subscription fails
here rather than as the integration's ``No SageMaker endpoints registered for
model`` on the first request.

Only API calls route through SageMaker. Chunk-size token counting uses the
vendored Kanon-2 *tokeniser* and never leaves the machine.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

API_KEY_ENV = "ISAACUS_API_KEY"  # pragma: allowlist secret -- env var name, not a key
ENDPOINTS_ENV = "ISAACUS_SAGEMAKER_ENDPOINTS"
REGION_ENV = "ISAACUS_SAGEMAKER_REGION"
PROFILE_ENV = "ISAACUS_SAGEMAKER_PROFILE"


@dataclass(frozen=True)
class SageMakerEndpoint:
    """One deployed SageMaker endpoint. ``models=None`` means "serves all"."""

    name: str
    models: tuple[str, ...] | None = None
    region: str | None = None

    def describe(self) -> str:
        served = "|".join(self.models) if self.models else "all models"
        return f"{self.name} ({served})"


def parse_endpoints(raw: str) -> list[SageMakerEndpoint]:
    """Parse an ``ISAACUS_SAGEMAKER_ENDPOINTS`` spec. Raises on malformed input.

    A malformed spec is loud rather than ignored: silently falling back to the
    hosted API would mean reaching for an API key that an air-gapped
    deployment does not have.
    """
    endpoints: list[SageMakerEndpoint] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        head, has_models, models_part = entry.partition("=")
        name, _, region = head.strip().partition("@")
        models = tuple(m.strip() for m in models_part.split("|") if m.strip())
        if not name.strip():
            raise ValueError(f"{ENDPOINTS_ENV}: entry {entry!r} has no endpoint name")
        if has_models and not models:
            raise ValueError(
                f"{ENDPOINTS_ENV}: endpoint {name.strip()!r} lists no models after "
                "'='; omit the '=' entirely if it serves every model"
            )
        endpoints.append(SageMakerEndpoint(
            name=name.strip(), models=models or None, region=region.strip() or None,
        ))
    return endpoints


def endpoints_from_env() -> list[SageMakerEndpoint]:
    """Endpoints declared in the environment; empty when hosted API is in use."""
    return parse_endpoints(os.environ.get(ENDPOINTS_ENV, ""))


def sagemaker_configured() -> bool:
    """True when the environment points Womblex at a SageMaker deployment."""
    return bool(endpoints_from_env())


#: Common misspellings of ``ISAACUS_SAGEMAKER_ENDPOINTS`` that silently fall
#: back to the hosted API. The singular form is the one deployers reach for
#: most (it reads naturally when a bundle is one endpoint), so a near-miss is
#: worth surfacing rather than leaving as a bare "No API key".
_ENDPOINTS_ALIASES: tuple[str, ...] = (
    "ISAACUS_SAGEMAKER_ENDPOINT",
    "ISAACUS_SAGEMAKER_ENDPT",
    "ISAACUS_SAGEMAKER_ENDPOINT_URL",
    "ISAACUS_ENDPOINT",
    "ISAACUS_ENDPOINTS",
    "ISAACUS_SAGE_MAKER_ENDPOINTS",
)


def misconfigured_endpoints_var() -> str | None:
    """A set env var that *looks like* the endpoints var but is not it.

    Returns the misspelled variable name (e.g. ``ISAACUS_SAGEMAKER_ENDPOINT``,
    singular) when one is set and the canonical ``ISAACUS_SAGEMAKER_ENDPOINTS``
    is not — the exact shape that reads as a bare "No API key" because the
    deployment silently falls back to the hosted API. ``None`` when the
    canonical var is set (nothing to warn about) or no near-miss is present.
    Cheap and network-free; used by the UI to turn a dead-end status into an
    actionable one.
    """
    if os.environ.get(ENDPOINTS_ENV, "").strip():
        return None
    for alias in _ENDPOINTS_ALIASES:
        if os.environ.get(alias, "").strip():
            return alias
    return None


def is_model_served(model: str, endpoints: Iterable[SageMakerEndpoint]) -> bool:
    """Whether some endpoint serves *model* (mirrors the integration's router)."""
    return any(e.models is None or model in e.models for e in endpoints)


def unserved_models(models: Iterable[str]) -> list[str]:
    """Of *models*, those no configured endpoint serves — empty on the hosted API."""
    endpoints = endpoints_from_env()
    if not endpoints:
        return []
    return [m for m in dict.fromkeys(models) if m and not is_model_served(m, endpoints)]


def make_isaacus_client(*, models: Iterable[str] = ()):  # type: ignore[no-untyped-def]
    """Construct an Isaacus client for whichever deployment is configured.

    ``models`` are the model ids the caller intends to use. On SageMaker they
    are checked against the declared endpoints up front; on the hosted API they
    are ignored (every model is reachable with a valid key).
    """
    endpoints = endpoints_from_env()
    if endpoints:
        return _sagemaker_client(endpoints, models)
    return _api_client()


def make_ai_chunking_client(chunking_model: str | None):  # type: ignore[no-untyped-def]
    """Client for semchunk's AI chunking, or ``None`` when it isn't enabled.

    Passed ``None``, semchunk builds its own client from ``ISAACUS_API_KEY`` —
    which an air-gapped SageMaker deployment does not have. Constructing it
    here routes AI chunking through whichever deployment is configured.
    """
    if not chunking_model:
        return None
    return make_isaacus_client(models=[chunking_model])


def _api_client():  # type: ignore[no-untyped-def]
    """Hosted-API client, stripping whitespace from the API key.

    The Isaacus SDK reads ``ISAACUS_API_KEY`` from the environment as-is, so a
    stray trailing newline (common when a key is pasted into a ``.env`` file on
    Windows) reaches httpx as an illegal header value and fails with a cryptic
    ``LocalProtocolError``. Falls back to the SDK default when unset.
    """
    import isaacus

    key = os.environ.get(API_KEY_ENV)
    if key is not None and key.strip() != key:
        return isaacus.Isaacus(api_key=key.strip())
    return isaacus.Isaacus()


def _sagemaker_client(endpoints: list[SageMakerEndpoint], models: Iterable[str]):  # type: ignore[no-untyped-def]
    import isaacus

    try:
        from isaacus_sagemaker import (
            IsaacusSageMakerRuntimeEndpoint,
            IsaacusSageMakerRuntimeHTTPClient,
        )
    except ImportError as e:  # pragma: no cover - only on a broken/partial install
        raise ImportError(
            f"{ENDPOINTS_ENV} is set but the isaacus-sagemaker package could not be "
            "imported. It is a core dependency; reinstall womblex."
        ) from e

    missing = unserved_models(models)
    if missing:
        raise ValueError(
            f"No SageMaker endpoint serves {', '.join(missing)}. {ENDPOINTS_ENV} "
            f"declares: {', '.join(e.describe() for e in endpoints)}. Subscribe to "
            "and deploy that model's Marketplace package and add its endpoint, or "
            "drop the '=model' restriction if an endpoint already serves it."
        )

    region = os.environ.get(REGION_ENV) or None
    profile = os.environ.get(PROFILE_ENV) or None
    _require_region(endpoints, region, profile)

    http_client = IsaacusSageMakerRuntimeHTTPClient(
        endpoints=[
            IsaacusSageMakerRuntimeEndpoint(
                name=e.name,
                region=e.region,
                models=list(e.models) if e.models else None,
            )
            for e in endpoints
        ],
        region=region,
        profile=profile,
    )
    logger.info(
        "Isaacus client: SageMaker deployment — %s",
        ", ".join(e.describe() for e in endpoints),
    )
    return isaacus.Isaacus(http_client=http_client)


def _require_region(
    endpoints: list[SageMakerEndpoint], region: str | None, profile: str | None,
) -> None:
    """Fail early when no region resolves for an endpoint.

    The integration asserts on the region only when building a request, which
    surfaces as a bare ``AssertionError`` mid-run (and is skipped entirely
    under ``python -O``). Resolving it here turns that into a fixable message.
    """
    if region is None and any(not e.region for e in endpoints):
        # Only consult AWS for a default when some endpoint actually needs one.
        import boto3

        try:
            region = boto3.Session(**({"profile_name": profile} if profile else {})).region_name
        except Exception as e:
            raise ValueError(f"Could not open an AWS session for SageMaker: {e}") from e
    unregioned = [e.name for e in endpoints if not (e.region or region)]
    if unregioned:
        raise ValueError(
            f"No AWS region for SageMaker endpoint(s) {', '.join(unregioned)}. Set "
            f"{REGION_ENV} (or AWS_REGION), or append '@<region>' to the entry in "
            f"{ENDPOINTS_ENV}."
        )


__all__ = [
    "ENDPOINTS_ENV",
    "PROFILE_ENV",
    "REGION_ENV",
    "SageMakerEndpoint",
    "endpoints_from_env",
    "is_model_served",
    "make_ai_chunking_client",
    "make_isaacus_client",
    "misconfigured_endpoints_var",
    "parse_endpoints",
    "sagemaker_configured",
    "unserved_models",
]
