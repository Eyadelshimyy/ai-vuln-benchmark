"""Health check and metrics endpoints."""

from __future__ import annotations

import structlog
from flask import Blueprint, current_app, jsonify
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)

logger = structlog.get_logger()

health_bp = Blueprint("health", __name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "search_service_requests_total",
    "Total number of requests to the search service",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "search_service_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
SEARCH_COUNT = Counter(
    "search_service_searches_total",
    "Total number of search queries executed",
)
INDEX_COUNT = Counter(
    "search_service_index_operations_total",
    "Total number of index operations",
    ["operation", "type"],
)


@health_bp.route("/health")
def health() -> tuple:
    """Liveness check — returns 200 if the process is running."""
    return jsonify({
        "status": "alive",
        "service": "search-service",
    }), 200


@health_bp.route("/health/ready")
def readiness() -> tuple:
    """Readiness check — returns 503 if MeiliSearch is unreachable."""
    search_service = current_app.config.get("SEARCH_SERVICE")

    healthy = False
    if search_service:
        healthy = search_service.ping()

    if healthy:
        return jsonify({"ready": True}), 200
    return jsonify({"ready": False, "reason": "meilisearch_unavailable"}), 503


@health_bp.route("/metrics")
def metrics() -> tuple:
    """Prometheus metrics endpoint."""
    return (
        generate_latest(),
        200,
        {"Content-Type": "text/plain; charset=utf-8"},
    )
