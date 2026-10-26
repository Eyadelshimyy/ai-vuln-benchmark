"""Seed module 09: Platform announcements (25+ entries)."""
import random
from datetime import timedelta

from helpers import (
    stable_id, ADMIN_USER_IDS, DEPARTMENTS,
    now, days_ago, hours_ago, bulk_insert, Json
)


def seed(cur, ns: str) -> int:
    """Insert 25+ announcements. Returns row count."""
    rng = random.Random(42)

    # ── Announcement definitions ──────────────────────────────────────────
    # Each tuple: (title, body, severity, status, target_depts, target_roles,
    #              starts_days_ago, ends_days_from_now_or_none, has_creator)
    raw = [
        # --- info / active ---
        (
            "New CI/CD pipeline dashboard now available",
            "The revamped CI/CD pipeline dashboard is live in the developer portal. "
            "You can now view build history, deployment frequency, and failure rates "
            "across all services. Bookmark it from the sidebar under Tooling.",
            "info", "active", ["DevOps", "Backend", "Frontend"], [], 3, 30, True,
        ),
        (
            "Platform latency improvements — P99 reduced by 40%",
            "After the Q2 infrastructure push we have reduced P99 API latency from "
            "210 ms to 125 ms across all public endpoints. Grafana dashboards have "
            "been updated with the new baseline thresholds.",
            "info", "active", [], [], 7, None, True,
        ),
        (
            "Welcome new SRE team members!",
            "Please welcome Anika Patel and Marcus Chen who are joining the SRE team "
            "this week. They will be ramping up on the on-call rotation over the next "
            "two sprints. Reach out and say hello!",
            "info", "active", ["SRE", "Infrastructure"], [], 2, 14, True,
        ),
        (
            "Kubernetes certification workshop — sign up now",
            "We are offering a sponsored CKA certification prep workshop starting "
            "July 10. Sessions run every Wednesday for six weeks. Register via the "
            "Learning Portal before June 30 to reserve your spot.",
            "info", "active", [], [], 5, 20, True,
        ),
        (
            "New Slack channel: #platform-announcements",
            "All platform-wide announcements will now also be mirrored to "
            "#platform-announcements on Slack. Subscribe to the channel to stay "
            "informed about maintenance windows and incident updates.",
            "info", "active", [], [], 10, None, True,
        ),
        (
            "Internal developer survey results published",
            "The results of the Q1 developer experience survey are now available in "
            "Confluence. Key takeaways include improved satisfaction with CI speed and "
            "requests for better staging environment parity.",
            "info", "active", [], [], 14, None, True,
        ),
        (
            "Library upgrades: React 19 and Angular 18 approved",
            "The frontend platform team has approved React 19 and Angular 18 for "
            "production use. Migration guides are available in the wiki. Please "
            "coordinate major version bumps with your tech lead.",
            "info", "active", ["Frontend", "Mobile", "Design"], [], 4, 45, True,
        ),
        (
            "OpenTelemetry tracing enabled for all services",
            "Distributed tracing via OpenTelemetry is now enabled across the service "
            "mesh. Traces are exported to Grafana Tempo and can be queried from the "
            "Explore tab. No code changes are needed for instrumented frameworks.",
            "info", "active", [], [], 6, None, True,
        ),
        # --- info / draft ---
        (
            "Upcoming: GitHub Copilot enterprise rollout",
            "We are planning to roll out GitHub Copilot Business licenses to all "
            "engineering staff in Q3. A pilot group of 50 developers will start in "
            "July. Stay tuned for enrollment details.",
            "info", "draft", [], [], None, None, True,
        ),
        (
            "Draft: New on-call compensation policy",
            "HR and Engineering leadership are finalizing a revised on-call "
            "compensation policy that includes tiered pay for weeknight and weekend "
            "pages. Review is expected to conclude by end of month.",
            "info", "draft", ["SRE", "DevOps", "Infrastructure"], [], None, None, True,
        ),
        (
            "Planned: Engineering all-hands Q3 kickoff",
            "The Q3 engineering all-hands is tentatively scheduled for July 15 at "
            "10:00 UTC. Agenda items include OKR review, architecture roadmap, and "
            "a demo showcase. Calendar invites will follow once confirmed.",
            "info", "draft", [], [], None, None, True,
        ),
        (
            "Draft: Shared component library v2 RFC open for review",
            "The Design Systems team has published an RFC for the next major version "
            "of the shared component library. Key changes include tree-shakeable "
            "exports and a CSS-in-JS migration. Comment on the RFC by July 5.",
            "info", "draft", ["Frontend", "Design", "Developer Experience"], [], None, None, True,
        ),
        (
            "Draft: GPU cluster access for ML workloads",
            "We are provisioning a dedicated GPU node pool in the dev cluster for "
            "ML training jobs. Access will be gated via resource quotas. Submit your "
            "team's projected usage to the ML Platform team by end of week.",
            "info", "draft", ["ML Engineering", "Data Science"], [], None, None, True,
        ),
        (
            "Grafana alerting migration to Alertmanager complete",
            "Legacy Grafana notification channels have been replaced by Alertmanager "
            "routes. All existing alert rules have been migrated. Review your team's "
            "routing configuration in the observability wiki.",
            "info", "active", ["SRE", "DevOps"], [], 8, None, True,
        ),
        (
            "New hire onboarding checklist updated",
            "The engineering onboarding checklist has been refreshed with updated "
            "links to environment setup guides, VPN instructions, and required "
            "compliance training modules. Share with your new team members.",
            "info", "active", [], [], 12, None, True,
        ),
        (
            "Draft: Annual tech radar review scheduled",
            "The annual technology radar review session is being planned for late "
            "July. Teams should prepare proposals for new technology adoptions and "
            "deprecation candidates. Template available in the engineering wiki.",
            "info", "draft", [], [], None, None, True,
        ),
        # --- info / expired ---
        (
            "Office network maintenance completed",
            "The scheduled office network maintenance on May 20 has been completed. "
            "All VPN tunnels and internal DNS resolvers are restored to normal "
            "operation. Report any lingering connectivity issues to IT.",
            "info", "expired", [], [], 45, -5, True,
        ),
        (
            "Hackathon 2025 results announced",
            "Congratulations to Team Kelp Forest for winning the spring hackathon "
            "with their real-time anomaly detection prototype. All project repos "
            "have been archived under the hackathon-2025 GitHub org.",
            "info", "expired", [], [], 60, -10, True,
        ),
        # --- warning / active ---
        (
            "Mandatory password rotation by end of quarter",
            "Per the updated security policy, all employees must rotate their SSO "
            "passwords before June 30. Accounts with passwords older than 90 days "
            "will be locked on July 1. Use the identity portal to update now.",
            "warning", "active", [], [], 15, 7, False,
        ),
        (
            "Legacy REST API v1 sunset on March 31",
            "REST API v1 will be permanently decommissioned on March 31. All "
            "consumers must migrate to v2 before then. The migration guide and "
            "compatibility shim are documented in the API wiki.",
            "warning", "active", ["Backend", "Platform Engineering"], [], 30, 60, True,
        ),
        (
            "Updated data retention policy effective next month",
            "The revised data retention policy takes effect on July 1. PII in "
            "non-production databases must be purged within 30 days of creation. "
            "Reach out to the Data Engineering team if you need exemptions.",
            "warning", "active", ["Data Engineering", "Data Science", "Security"], [], 10, 25, True,
        ),
        (
            "SOC 2 audit preparation — action items for all teams",
            "Our annual SOC 2 Type II audit begins August 1. Every team must "
            "verify that access reviews are current, runbooks are up to date, and "
            "change management tickets reference the correct approval chain.",
            "warning", "active", [], ["admin"], 5, 40, True,
        ),
        (
            "Staging environment instability — use with caution",
            "The shared staging cluster is experiencing intermittent pod evictions "
            "due to memory pressure. The infra team is adding capacity this week. "
            "Avoid running load tests in staging until further notice.",
            "warning", "active", ["Backend", "QA", "SRE"], [], 1, 5, True,
        ),
        # --- warning / draft ---
        (
            "Draft: Deprecation of internal PyPI mirror",
            "We are planning to deprecate the self-hosted PyPI mirror in favor of "
            "Artifactory. A migration timeline will be shared once the Artifactory "
            "instance passes security review.",
            "warning", "draft", ["Data Engineering", "ML Engineering", "Backend"], [], None, None, True,
        ),
        (
            "Planned: Terraform state migration to S3 backend",
            "Infrastructure modules currently using local Terraform state will be "
            "migrated to a shared S3 backend with DynamoDB locking. Affected teams "
            "will be contacted individually with migration steps.",
            "warning", "draft", ["Infrastructure", "DevOps", "SRE"], [], None, None, True,
        ),
        # --- warning / expired ---
        (
            "VPN certificate renewal deadline passed",
            "The deadline to renew VPN client certificates was May 15. All "
            "certificates have been reissued automatically. If you are unable to "
            "connect, re-download your profile from the identity portal.",
            "warning", "expired", [], [], 50, -8, False,
        ),
        # --- critical / active ---
        (
            "Database migration scheduled for Saturday 02:00-04:00 UTC",
            "A critical PostgreSQL schema migration will run this Saturday from "
            "02:00 to 04:00 UTC. Expect up to 15 minutes of read-only mode on the "
            "primary cluster. All non-essential batch jobs will be paused.",
            "critical", "active", ["Backend", "Data Engineering", "SRE"], [], 1, 3, True,
        ),
        (
            "Post-mortem: June 15 outage — S3 connectivity",
            "The root cause of the 47-minute outage on June 15 was an expired IAM "
            "role trust policy for the file-service. Remediation includes automated "
            "policy expiry alerts and a secondary credential path. Full post-mortem "
            "is linked in the incident channel.",
            "critical", "active", [], [], 8, 30, True,
        ),
        (
            "Security incident: compromised npm dependency detected",
            "A supply-chain compromise was detected in event-stream@4.1.2 used by "
            "the notification service. The package has been pinned to a safe version "
            "and all affected containers have been redeployed. Scan your lockfiles.",
            "critical", "active", ["Security", "Backend", "Frontend"], [], 2, 14, False,
        ),
        # --- critical / draft ---
        (
            "Draft: Emergency maintenance — Redis cluster failover",
            "We may need to perform an emergency failover of the primary Redis "
            "cluster due to increasing replication lag. Impact assessment and "
            "rollback plan are under review. ETA for decision: 24 hours.",
            "critical", "draft", ["SRE", "Backend", "Infrastructure"], [], None, None, True,
        ),
        # --- critical / expired ---
        (
            "Resolved: Elevated error rates on auth-service",
            "The elevated 5xx error rates on auth-service between 14:00 and 15:30 "
            "UTC on May 28 have been resolved. A misconfigured rate-limiter was "
            "throttling legitimate traffic. The config has been corrected.",
            "critical", "expired", [], [], 35, -3, True,
        ),
        (
            "Resolved: Data pipeline backlog cleared",
            "The Kafka consumer lag that peaked at 2.4M messages on May 22 has been "
            "fully drained. Root cause was an under-provisioned consumer group after "
            "the partition rebalance. Auto-scaling rules have been tightened.",
            "critical", "expired", [], [], 40, -6, False,
        ),
    ]

    columns = [
        "id", "title", "body", "severity", "status",
        "target_audience", "starts_at", "ends_at", "created_by",
        "created_at", "updated_at",
    ]

    rows = []
    for i, (title, body, severity, status, depts, roles,
            start_days, end_offset, has_creator) in enumerate(raw):

        uid = stable_id("announcement", i)

        # target_audience
        audience = {}
        if depts:
            audience["departments"] = depts
        if roles:
            audience["roles"] = roles

        # starts_at / ends_at
        if start_days is not None:
            starts_at = days_ago(start_days)
        else:
            starts_at = None

        if end_offset is not None and starts_at is not None:
            ends_at = now() + timedelta(days=end_offset)
        else:
            ends_at = None

        created_by = rng.choice(ADMIN_USER_IDS) if has_creator else None

        # created_at: slightly before starts_at, or random recent date for drafts
        if starts_at is not None:
            created_at = starts_at - timedelta(hours=rng.randint(1, 48))
        else:
            created_at = days_ago(rng.randint(1, 14))
        updated_at = created_at + timedelta(hours=rng.randint(0, 12))

        rows.append((
            uid, title, body, severity, status,
            Json(audience), starts_at, ends_at, created_by,
            created_at, updated_at,
        ))

    template = "(%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)"
    count = bulk_insert(cur, "announcements", columns, rows,
                        on_conflict="DO NOTHING", template=template)
    return count
