"""Seed module 10: Incidents (50+) and system configs (30+)."""
import random
from datetime import timedelta
from helpers import (
    stable_id, ADMIN_USER_IDS,
    now, days_ago, hours_ago, bulk_insert, Json
)

SERVICES = [
    "api-gateway", "auth-service", "file-service", "document-service",
    "collab-service", "notification-service", "admin-service",
    "analytics-service", "audit-service", "search-service", "report-service",
]

SEVERITIES = ["low"] * 30 + ["medium"] * 35 + ["high"] * 25 + ["critical"] * 10
STATUSES = ["open"] * 15 + ["investigating"] * 20 + ["resolved"] * 40 + ["closed"] * 25
DEVIN_SESSION_STATUSES = ["completed", "in_progress", "failed"]

INCIDENT_TEMPLATES = [
    ("API Gateway 502 errors spike",
     "Upstream services returning 502 Bad Gateway intermittently. "
     "Load balancer health checks passing but individual requests timing out after 30s. "
     "Error rate peaked at 12% of total traffic during the incident window.",
     "api-gateway"),
    ("Redis connection pool exhaustion",
     "Redis connection pool hit max capacity of 256 connections. "
     "Clients blocking on connection acquisition with 5s timeout. "
     "Caused cascading latency in session lookups and rate limiting.",
     "auth-service"),
    ("S3 upload failures in us-east-1",
     "PutObject calls to the primary upload bucket returning 503 SlowDown errors. "
     "AWS reporting elevated error rates in us-east-1. "
     "Retry logic with exponential backoff mitigating but not eliminating failures.",
     "file-service"),
    ("Document service OOM kills",
     "Kubernetes OOMKilled events on document-service pods processing large DOCX conversions. "
     "Memory usage spiking to 4Gi limit during concurrent PDF renders. "
     "Horizontal pod autoscaler unable to scale fast enough.",
     "document-service"),
    ("Search index corruption after reindex",
     "Full reindex job corrupted the primary search index shard. "
     "Queries returning stale or missing results for documents updated in the last 48h. "
     "Fallback to replica shard restored partial service.",
     "search-service"),
    ("Auth service JWT validation failures",
     "JWT signature verification failing for tokens issued before the last key rotation. "
     "JWKS endpoint returning the new key but clients caching the old public key. "
     "Approximately 8% of authenticated requests affected.",
     "auth-service"),
    ("Database connection leak in file-service",
     "PostgreSQL connection count growing linearly over 72h. "
     "Connection pool not reclaiming idle connections due to misconfigured maxLifetime. "
     "Database approaching max_connections limit of 200.",
     "file-service"),
    ("CI/CD pipeline stuck — GitHub webhook backlog",
     "GitHub webhook delivery queue backed up with 3,400 pending events. "
     "Pipeline triggers delayed by 25+ minutes. "
     "Root cause traced to a webhook endpoint returning 429 due to rate limiting.",
     "admin-service"),
    ("Grafana dashboard 503 errors",
     "Grafana returning 503 Service Unavailable on all dashboard loads. "
     "Underlying Prometheus data source timing out on range queries spanning >7d. "
     "Restarting Grafana pods temporarily restored access.",
     "analytics-service"),
    ("DynamoDB throttling on metadata writes",
     "WriteCapacityUnits exceeded on the file-metadata table during bulk import. "
     "Provisioned capacity at 500 WCU but burst traffic hitting 2,000 WCU. "
     "Switching to on-demand capacity mode resolved throttling.",
     "file-service"),
    ("Certificate expiry — collab-service TLS",
     "TLS certificate for collab-service internal endpoint expired. "
     "Mutual TLS handshake failures between collab-service and api-gateway. "
     "Service mesh sidecar logs showing x509 certificate has expired errors.",
     "collab-service"),
    ("Memory leak in notification-service",
     "Heap usage in notification-service growing 50MB/hour without recovery. "
     "GC logs showing full GC pauses exceeding 2s every 15 minutes. "
     "Profiling identified unbounded cache of rendered email templates.",
     "notification-service"),
    ("Slow queries on audit_logs table",
     "SELECT queries on audit_logs exceeding 30s for time-range filters. "
     "Table grown to 180M rows with missing index on created_at column. "
     "Adding a BRIN index reduced p99 query time from 32s to 120ms.",
     "audit-service"),
    ("Cross-site scripting vulnerability in admin dashboard",
     "Reflected XSS found in the admin search query parameter. "
     "User-supplied input rendered without sanitization in error messages. "
     "Severity escalated after proof-of-concept demonstrated session hijacking.",
     "admin-service"),
    ("Data loss in analytics pipeline",
     "ETL job silently dropping events with null user_id field. "
     "Approximately 4% of analytics events lost over a 2-week window. "
     "Root cause was a schema change that made user_id non-nullable without backfill.",
     "analytics-service"),
    ("Notification delivery delays exceeding SLA",
     "Email and push notifications delayed by 45+ minutes during peak hours. "
     "SQS queue depth growing due to insufficient consumer concurrency. "
     "Consumer lambda concurrency limit raised from 10 to 50.",
     "notification-service"),
    ("Collab-service WebSocket disconnections",
     "Users experiencing frequent WebSocket disconnects during collaborative editing. "
     "ALB idle timeout set to 60s causing drops for inactive but open sessions. "
     "Ping/pong keepalive interval adjusted from 120s to 30s.",
     "collab-service"),
    ("Report generation timeout for large datasets",
     "Report-service timing out when generating exports with >100k rows. "
     "Single-threaded CSV serialization consuming 100% CPU for 5+ minutes. "
     "Streaming response with chunked transfer encoding resolved the timeout.",
     "report-service"),
    ("Admin dashboard RBAC bypass",
     "Non-admin users able to access restricted endpoints via direct URL manipulation. "
     "Server-side authorization check missing on three admin API routes. "
     "Hotfix deployed within 2h of discovery; no evidence of exploitation.",
     "admin-service"),
    ("Search service latency spike during peak traffic",
     "p99 search latency increased from 200ms to 4.5s during morning peak. "
     "OpenSearch cluster running hot with JVM heap pressure above 85%. "
     "Adding two data nodes and rebalancing shards restored performance.",
     "search-service"),
    ("File versioning race condition",
     "Concurrent uploads to the same file creating orphaned versions. "
     "Optimistic locking check using stale version number in 0.3% of cases. "
     "Added SELECT FOR UPDATE to the version increment path.",
     "file-service"),
    ("API gateway request routing misconfiguration",
     "Traffic to /api/v2/documents routed to the legacy v1 handler after deploy. "
     "Ingress controller config overwritten by Helm chart upgrade. "
     "Rollback of ingress resource restored correct routing within 8 minutes.",
     "api-gateway"),
    ("Auth token refresh loop",
     "Clients entering infinite refresh loop when access token and refresh token expire simultaneously. "
     "Race condition in token refresh middleware causing 401 cascade. "
     "Fix added a grace period of 30s before refresh token hard expiry.",
     "auth-service"),
    ("Audit service write amplification",
     "Audit event inserts causing 10x write amplification due to over-indexing. "
     "INSERT latency p99 at 800ms blocking upstream request completion. "
     "Dropped three unused indexes reducing write latency to 15ms.",
     "audit-service"),
    ("Document conversion service crash loop",
     "LibreOffice soffice process crashing on malformed .pptx files. "
     "Kubernetes CrashLoopBackOff with 5-minute restart penalty. "
     "Added input validation and a 60s conversion timeout per document.",
     "document-service"),
    ("Analytics event schema drift",
     "Frontend SDK sending events with renamed fields after a release. "
     "Analytics pipeline rejecting 22% of incoming events as schema violations. "
     "Schema registry updated and a backfill job re-processed dropped events.",
     "analytics-service"),
    ("API gateway rate limiter false positives",
     "Legitimate API clients being rate-limited due to shared IP behind corporate proxy. "
     "Rate limiter keyed on X-Forwarded-For which collapsed to a single IP. "
     "Switched rate limit key to authenticated user ID for logged-in clients.",
     "api-gateway"),
    ("Notification template rendering failure",
     "Handlebars template engine throwing on undefined variable in welcome email. "
     "New user registrations not receiving confirmation emails for 6h. "
     "Default values added to all template variables as a safeguard.",
     "notification-service"),
    ("File service presigned URL expiry too short",
     "Users receiving 403 Forbidden when downloading files shared via link. "
     "Presigned URL TTL set to 5 minutes but large files taking 10+ minutes to download. "
     "Increased presigned URL expiry to 1h with content-disposition enforcement.",
     "file-service"),
    ("Collab-service CRDT merge conflict",
     "Yjs document state diverging between two concurrent editors. "
     "Root cause traced to a clock skew issue in the awareness protocol. "
     "Deploying a centralized document state server resolved merge anomalies.",
     "collab-service"),
    ("Admin service background job deadlock",
     "Sidekiq workers deadlocked on competing database row locks during nightly cleanup. "
     "Job queue backed up to 12k pending jobs by morning. "
     "Refactored cleanup to use advisory locks and smaller batch sizes.",
     "admin-service"),
    ("Search indexing lag exceeding 30 minutes",
     "Document updates not appearing in search results for 30+ minutes. "
     "Indexing consumer falling behind due to large payload deserialization overhead. "
     "Switched to a binary protocol and increased consumer batch size to 500.",
     "search-service"),
    ("Report service PDF rendering blank pages",
     "PDF exports containing blank pages for documents with embedded SVG charts. "
     "Headless Chrome renderer failing silently on SVG foreignObject elements. "
     "Replaced SVG rendering path with server-side canvas rasterization.",
     "report-service"),
    ("Auth service LDAP sync failure",
     "LDAP directory sync job failing with connection timeout to corporate AD. "
     "New user provisioning blocked for 18h until manual sync triggered. "
     "Added retry with fallback to cached directory snapshot.",
     "auth-service"),
    ("API gateway memory leak under load",
     "Go runtime heap growing unboundedly under sustained 5k req/s. "
     "Goroutine leak in the middleware chain not cancelling context on client disconnect. "
     "Fix propagated context cancellation through the entire handler chain.",
     "api-gateway"),
    ("Analytics service Kafka consumer lag",
     "Consumer group falling behind by 2M messages during reprocessing window. "
     "Partition rebalance storm caused by aggressive session.timeout.ms of 6s. "
     "Increased timeout to 30s and pinned partition assignment.",
     "analytics-service"),
    ("File service thumbnail generation backlog",
     "Thumbnail generation queue depth at 45k with 6h estimated drain time. "
     "Image processing pods CPU-throttled at 500m limit. "
     "Temporarily scaled to 10 replicas and increased CPU limit to 2 cores.",
     "file-service"),
    ("Document service concurrent edit limit reached",
     "Maximum of 25 concurrent editors per document hit on the Q3 planning doc. "
     "Users receiving 429 Too Many Requests when joining the editing session. "
     "Raised limit to 50 after load testing confirmed server capacity.",
     "document-service"),
    ("Notification service SMS gateway errors",
     "Twilio API returning 21608 (unverified destination) for international numbers. "
     "SMS notifications failing for 15% of users with non-US phone numbers. "
     "Registered additional sender IDs for EU and APAC regions.",
     "notification-service"),
    ("Audit log export CSV injection",
     "Audit log export including unsanitized user input in CSV cells. "
     "Potential formula injection via fields starting with = or @. "
     "Added cell value escaping with single-quote prefix for special characters.",
     "audit-service"),
    ("Admin dashboard SSO integration failure",
     "SAML assertion validation failing after IdP certificate rotation. "
     "Admin users unable to log in for 3h until new certificate fingerprint added. "
     "Implemented automatic certificate refresh from IdP metadata URL.",
     "admin-service"),
    ("Search service query injection vulnerability",
     "OpenSearch query DSL injection via unsanitized search terms. "
     "Specially crafted query strings could access unauthorized index data. "
     "Input sanitization and parameterized queries deployed as emergency fix.",
     "search-service"),
    ("API gateway TLS 1.0 deprecation breakage",
     "Legacy clients using TLS 1.0 failing to connect after security hardening deploy. "
     "Approximately 2% of API traffic affected, primarily from older mobile SDKs. "
     "Added TLS 1.2 minimum enforcement with a 30-day grace period for migration.",
     "api-gateway"),
    ("Collab-service presence tracking stale users",
     "User presence indicators showing users as online hours after disconnecting. "
     "Redis TTL on presence keys not being refreshed on WebSocket close. "
     "Added explicit key deletion in the WebSocket onClose handler.",
     "collab-service"),
    ("Report service memory spike on pivot tables",
     "Pivot table aggregation loading entire dataset into memory for large reports. "
     "Pod OOMKilled generating a report with 500k row source data. "
     "Refactored to streaming aggregation with database-side GROUP BY.",
     "report-service"),
    ("File service virus scan queue backup",
     "ClamAV scan queue growing unboundedly due to a hung scanner process. "
     "New uploads quarantined for 4h awaiting scan completion. "
     "Added a 120s scan timeout and automatic process recycling.",
     "file-service"),
    ("Auth service brute force detection gaps",
     "Rate limiting on login endpoint only counting successful auth attempts. "
     "Brute force attacks generating 10k failed attempts/minute unthrottled. "
     "Updated rate limiter to count all attempts regardless of outcome.",
     "auth-service"),
    ("Document service LaTeX rendering RCE",
     "LaTeX input allowing arbitrary command execution via \\input and \\write18. "
     "Sandbox escape possible through crafted .tex document uploads. "
     "Disabled shell-escape and restricted \\input to whitelisted paths.",
     "document-service"),
    ("Analytics dashboard data inconsistency",
     "Dashboard showing different totals depending on selected time granularity. "
     "Pre-aggregated hourly and daily rollup tables diverging by 3-5%. "
     "Root cause was a timezone conversion bug in the rollup cron job.",
     "analytics-service"),
    ("Notification service webhook retry storm",
     "Failed webhook deliveries retrying every 5s without backoff for 48h. "
     "Target endpoint down causing 1.2M retry attempts and elevated SQS costs. "
     "Implemented exponential backoff with a maximum of 5 retry attempts.",
     "notification-service"),
    ("Admin service bulk user import timeout",
     "CSV import of 10k users timing out at the 30s API gateway limit. "
     "Synchronous processing blocking the request thread for the entire import. "
     "Converted to async job with progress polling endpoint.",
     "admin-service"),
    ("Collab-service document merge data loss",
     "Concurrent offline edits resulting in lost paragraphs after reconnection. "
     "CRDT vector clock not advancing correctly for offline-buffered operations. "
     "Patch applied to flush pending operations before initiating sync handshake.",
     "collab-service"),
    ("Search service autocomplete returning deleted items",
     "Autocomplete suggestions including soft-deleted documents and files. "
     "Suggestion index not subscribing to delete events from the event bus. "
     "Added delete event consumer and purged stale entries from the suggestion index.",
     "search-service"),
    ("API gateway health check false healthy",
     "Health endpoint returning 200 while downstream services are unreachable. "
     "Shallow health check only verifying the gateway process, not dependencies. "
     "Implemented deep health check with circuit breaker status for each upstream.",
     "api-gateway"),
]


def seed(cur, ns: str) -> int:
    """Insert incidents and system configs. Returns total row count."""
    count = 0
    count += _seed_incidents(cur)
    count += _seed_system_configs(cur)
    return count


def _seed_incidents(cur) -> int:
    rng = random.Random(1010)

    columns = [
        "id", "title", "description", "severity", "status",
        "affected_service", "devin_session_id", "devin_session_url",
        "devin_session_status", "reporter_id", "resolved_at",
        "created_at", "updated_at", "closed_at",
    ]

    rows = []
    for i, (title, description, service) in enumerate(INCIDENT_TEMPLATES):
        severity = rng.choice(SEVERITIES)
        status = rng.choice(STATUSES)

        created_at = days_ago(rng.randint(1, 180))
        updated_at = created_at + timedelta(hours=rng.randint(1, 72))

        resolved_at = None
        closed_at = None
        if status in ("resolved", "closed"):
            resolved_at = created_at + timedelta(hours=rng.randint(1, 48))
        if status == "closed":
            base = resolved_at or created_at
            closed_at = base + timedelta(hours=rng.randint(1, 24))

        devin_session_id = None
        devin_session_url = None
        devin_session_status = None
        if rng.random() < 0.30:
            hex_part = stable_id("devin-session", i).replace("-", "")[:12]
            devin_session_id = f"session-{hex_part}"
            devin_session_url = f"https://app.devin.ai/sessions/{devin_session_id}"
            devin_session_status = rng.choice(DEVIN_SESSION_STATUSES)

        reporter_id = rng.choice(ADMIN_USER_IDS)

        rows.append((
            stable_id("incident", i),
            title,
            description,
            severity,
            status,
            service,
            devin_session_id,
            devin_session_url,
            devin_session_status,
            reporter_id,
            resolved_at,
            created_at,
            updated_at,
            closed_at,
        ))

    return bulk_insert(cur, "incidents", columns, rows, on_conflict="DO NOTHING")


def _seed_system_configs(cur) -> int:
    rng = random.Random(1011)

    configs = [
        ("max_upload_size_bytes", "10737418240", "integer", "Maximum file upload size in bytes (10 GB)", False),
        ("session_timeout_minutes", "30", "integer", "User session idle timeout", False),
        ("max_concurrent_uploads", "10", "integer", "Maximum simultaneous file uploads per user", False),
        ("default_storage_tier", "free", "string", "Default storage tier for new organizations", False),
        ("maintenance_mode", "false", "boolean", "Global maintenance mode toggle", False),
        ("api_rate_limit_per_minute", "1000", "integer", "API rate limit per authenticated user per minute", False),
        ("search_index_refresh_interval_seconds", "30", "integer", "Interval between search index refreshes", False),
        ("max_file_version_count", "50", "integer", "Maximum number of versions retained per file", False),
        ("audit_log_retention_days", "365", "integer", "Number of days to retain audit log entries", False),
        ("backup_schedule_cron", "0 2 * * *", "string", "Daily backup schedule in cron syntax (02:00 UTC)", False),
        ("smtp_host", "smtp.otterworks.io", "string", "SMTP relay hostname for outbound email", False),
        ("redis_max_connections", "100", "integer", "Maximum Redis connection pool size", False),
        ("cors_allowed_origins", "https://app.otterworks.io,https://admin.otterworks.io", "string",
         "Comma-separated list of allowed CORS origins", False),
        ("feature_flag_cache_ttl_seconds", "60", "integer", "TTL for feature flag evaluation cache", False),
        ("max_team_members", "500", "integer", "Maximum members per team", False),
        ("webhook_retry_max_attempts", "5", "integer", "Maximum retry attempts for failed webhook deliveries", False),
        ("webhook_retry_backoff_seconds", "30", "integer", "Base backoff interval between webhook retries", False),
        ("oauth_token_expiry_seconds", "3600", "integer", "OAuth2 access token lifetime", False),
        ("encryption_algorithm", "AES-256-GCM", "string", "Default encryption algorithm for data at rest", False),
        ("database_pool_size", "20", "integer", "PostgreSQL connection pool size per service instance", False),
        ("log_level", "info", "string", "Global application log level", False),
        ("sentry_dsn", "https://***@sentry.otterworks.io/1", "string",
         "Sentry error tracking DSN", True),
        ("slack_webhook_url", "https://hooks.slack.com/services/***", "string",
         "Slack integration webhook URL for alerts", True),
        ("aws_region", "us-east-1", "string", "Primary AWS region for infrastructure", False),
        ("cdn_base_url", "https://cdn.otterworks.io", "string", "CDN base URL for static asset delivery", False),
        ("max_document_size_bytes", "104857600", "integer", "Maximum document size in bytes (100 MB)", False),
        ("password_min_length", "12", "integer", "Minimum password length for user accounts", False),
        ("mfa_enforcement", "optional", "string", "MFA enforcement policy: disabled, optional, required", False),
        ("file_scan_enabled", "true", "boolean", "Enable virus scanning for uploaded files", False),
        ("max_api_key_count", "10", "integer", "Maximum API keys per user", False),
        ("jwt_signing_algorithm", "RS256", "string", "JWT signing algorithm", False),
        ("datadog_api_key", "dd-***-otterworks", "string", "Datadog API key for metrics export", True),
        ("pagerduty_integration_key", "pd-***-otterworks", "string",
         "PagerDuty integration key for incident routing", True),
        ("s3_bucket_primary", "otterworks-files-prod", "string", "Primary S3 bucket for file storage", False),
        ("max_search_results", "200", "integer", "Maximum results returned per search query", False),
        ("websocket_max_connections_per_doc", "50", "integer",
         "Maximum concurrent WebSocket connections per document", False),
        ("email_from_address", "noreply@otterworks.io", "string", "Default sender address for system emails", False),
        ("stripe_api_key", "sk_live_***", "string", "Stripe API key for billing integration", True),
        ("github_app_private_key", "-----BEGIN RSA PRIVATE KEY-----\\n***", "string",
         "GitHub App private key for CI/CD integration", True),
        ("idle_session_cleanup_minutes", "1440", "integer",
         "Idle session cleanup interval in minutes (24h)", False),
    ]

    columns = [
        "id", "key", "value", "value_type", "description", "is_secret",
        "created_at", "updated_at",
    ]

    rows = []
    for i, (key, value, value_type, description, is_secret) in enumerate(configs):
        created_at = days_ago(rng.randint(30, 365))
        updated_at = created_at + timedelta(hours=rng.randint(0, 720))

        rows.append((
            stable_id("system-config", i),
            key,
            value,
            value_type,
            description,
            is_secret,
            created_at,
            updated_at,
        ))

    return bulk_insert(cur, "system_configs", columns, rows,
                       on_conflict="(key) DO NOTHING")
