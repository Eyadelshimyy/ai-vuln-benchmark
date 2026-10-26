"""Seed module 08: Feature flags for engineering org (35+ flags)."""
import random
from datetime import timedelta

from helpers import (
    stable_id, ADMIN_USER_IDS, DEPARTMENTS, TEAMS,
    now, days_ago, bulk_insert, Json
)

FLAGS = [
    ("dark_mode_v2", "Toggle dark mode V2 UI across web and mobile clients"),
    ("realtime_collab", "Enable real-time collaborative editing via CRDT sync"),
    ("ai_code_review", "AI-powered code review suggestions on pull requests"),
    ("new_search_engine", "Switch to the new full-text search backend"),
    ("graphql_api", "Expose the GraphQL API layer for external integrations"),
    ("kubernetes_autoscale", "Enable HPA-based autoscaling for all deployments"),
    ("observability_dashboard_v3", "Rollout of the redesigned observability dashboard"),
    ("sso_okta_integration", "Okta SSO provider support for enterprise tenants"),
    ("webhook_retry_v2", "Exponential backoff retry logic for webhook delivery"),
    ("batch_export", "Allow bulk CSV/JSON export of workspace data"),
    ("mobile_push_notifications", "Push notification support for iOS and Android"),
    ("advanced_rbac", "Granular role-based access control with custom roles"),
    ("audit_log_export", "Export audit logs to external SIEM integrations"),
    ("ai_document_summary", "AI-generated summaries for uploaded documents"),
    ("file_versioning_v2", "Improved file version history with diff viewer"),
    ("custom_workflows", "User-defined approval and review workflows"),
    ("api_rate_limiting_v2", "Token-bucket rate limiting with per-tenant quotas"),
    ("elasticsearch_migration", "Migrate search index from Solr to Elasticsearch"),
    ("redis_cache_layer", "Enable Redis caching layer for hot-path queries"),
    ("ci_cd_pipeline_v2", "Next-gen CI/CD pipeline with parallel stage execution"),
    ("canary_deployments", "Canary deployment strategy for production rollouts"),
    ("feature_flag_analytics", "Track flag evaluation metrics and exposure events"),
    ("user_impersonation", "Allow admins to impersonate users for debugging"),
    ("bulk_user_import", "Bulk user provisioning via CSV upload"),
    ("slack_integration_v2", "Bidirectional Slack integration with thread sync"),
    ("teams_integration", "Microsoft Teams notifications and bot commands"),
    ("jira_sync", "Two-way sync between internal tasks and Jira issues"),
    ("github_pr_linking", "Auto-link GitHub PRs to internal documents"),
    ("automated_backups_v2", "Incremental backup strategy with point-in-time restore"),
    ("data_retention_policies", "Configurable data retention and auto-purge rules"),
    ("gdpr_export_tool", "Self-service GDPR data export for end users"),
    ("incident_auto_escalation", "Auto-escalate unacknowledged incidents after SLA breach"),
    ("chaos_engineering_toggle", "Enable chaos experiments in staging environments"),
    ("performance_profiler", "In-app performance profiling with flame graphs"),
    ("distributed_tracing_ui", "Unified distributed tracing viewer across services"),
    ("smart_notifications", "ML-based notification batching and prioritization"),
    ("live_query_inspector", "Real-time SQL query plan inspector for debugging"),
    ("service_mesh_migration", "Migrate inter-service traffic to Istio service mesh"),
    ("workspace_templates", "Pre-built workspace templates for quick onboarding"),
    ("email_digest_v2", "Redesigned daily/weekly email digest with personalization"),
]


def seed(cur, ns: str) -> int:
    """Insert 35+ feature flags. Returns row count."""
    rng = random.Random(42)
    _now = now()

    dept_lower = [d.lower().replace(" ", "_") for d in DEPARTMENTS]
    team_lower = [t.lower().replace(" ", "_").replace("/", "_") for t in TEAMS]

    rows = []
    for i, (name, description) in enumerate(FLAGS):
        enabled = rng.random() < 0.60

        if enabled:
            rollout_pct = rng.choice([10, 20, 25, 30, 50, 75, 80, 90, 100])
        else:
            rollout_pct = 0

        num_target_users = rng.randint(0, 5)
        target_users = rng.sample(ADMIN_USER_IDS, num_target_users)

        num_groups = rng.randint(0, 3)
        target_groups = rng.sample(
            dept_lower + team_lower, min(num_groups, len(dept_lower + team_lower))
        )

        expires_roll = rng.random()
        if expires_roll < 0.70:
            expires_at = None
        elif expires_roll < 0.90:
            expires_at = _now + timedelta(days=rng.randint(7, 120))
        else:
            expires_at = _now - timedelta(days=rng.randint(1, 60))

        created_at = _now - timedelta(
            days=rng.randint(0, 180),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        updated_at = created_at + timedelta(
            days=rng.randint(0, max((_now - created_at).days, 1)),
            hours=rng.randint(0, 12),
        )
        if updated_at > _now:
            updated_at = _now

        rows.append((
            stable_id("feature-flag", i),
            name,
            description,
            enabled,
            Json(target_users),
            Json(target_groups),
            rollout_pct,
            expires_at,
            created_at,
            updated_at,
        ))

    columns = [
        "id", "name", "description", "enabled",
        "target_users", "target_groups", "rollout_percentage",
        "expires_at", "created_at", "updated_at",
    ]

    count = bulk_insert(
        cur, "feature_flags", columns, rows,
        on_conflict="(name) DO NOTHING",
        template="(%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)",
    )
    return count
