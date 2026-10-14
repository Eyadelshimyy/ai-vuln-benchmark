"""Domain map for the rox-core root schema: which tables belong to which domain.

Rules are tried in order; the first matching pattern assigns the table.
Every table at the pinned commit must be either excluded or assigned.
"""

import re

EXCLUDED = re.compile(
    r"^(_schema_router_\w+_test|shim_\w+_probe|rox_db_test_probe|animal|alembic_version)$"
)

# (id, title, pattern, key tables shown as rows on the root card)
DOMAINS: list[tuple[str, str, str, list[str]]] = [
    (
        "tasks",
        "Background tasks, jobs & metering",
        r"cost_meter|^(task_|user_task|rox_task|api_jobs|digest|invoice_task"
        r"|data_processing|data_extraction|extraction_task|generic_backfill"
        r"|databricks|priorities|schema_router)",
        [
            "task_run",
            "user_task_run",
            "api_jobs",
            "extraction_task_status",
            "cost_meter_log",
            "generic_backfill_run",
        ],
    ),
    (
        "deals",
        "Deals, quotes & contracts",
        r"^(deal|product|quote|contract)",
        ["deal", "deal_person", "deal_stage", "contract", "quote", "product"],
    ),
    (
        "gov",
        "Governance & permissions",
        r"^(security_entity|permission|profile_permission|pod|record_access"
        r"|resource_access|governance|governed|org_wide_default|validation_rule"
        r"|team|provisioning_profile|rox_organization_user_roles|impersonation)",
        [
            "security_entity",
            "permission_set",
            "permission_set_assignment",
            "profile_permission",
            "pod",
            "team",
        ],
    ),
    (
        "crm",
        "CRM mirror (entity_*, sfdc_*, graph)",
        r"^(entity_|external_entity|external_column|sfdc_|crm_|graph|erd_"
        r"|unified_catalog|resolved_entity|archived_entity|data_controller"
        r"|relationships$|explicit_name)",
        [
            "entity_company",
            "entity_person",
            "entity_deal",
            "external_column_mapping",
            "crm_field",
            "graph",
        ],
    ),
    (
        "tenancy",
        "Tenancy: orgs, users, auth & billing",
        r"^(provider_credit|organization|org_extra|org_secret|org_default"
        r"|sub_organization|user$|user_extra|user_profile|user_preferences"
        r"|user_platform|auth0|api_key|plan$|subscription|enterprise_sub"
        r"|user_limit|payment_|purchased_|starter_action|auto_reload|downgrade"
        r"|delete_user|usage_notification|hierarchical_setting|workspace_setup"
        r"|feature_waitlist|mcp_org|push_device|rox_customer$|rox_feature_actions"
        r"|remaining_agents|model_token_rate|chat_user_preferences|user_account"
        r"|email_screening_rules$)",
        [
            "organization",
            "user",
            "user_profile",
            "sub_organization",
            "subscription",
            "auth0_organization",
        ],
    ),
    (
        "people",
        "Companies & people",
        r"^(org_chart|org_public_person|org_sender_mapping|account_|rox_company"
        r"|company|rox_enriched|rox_person|person|rox_lead|contact_|employee"
        r"|byok|clay|mixrank|proxycurl|rocket_reach|people_|linkedin_|crustdata"
        r"|zoominfo|perplexity|do_not_call|marketing_leads|rox_user_selected"
        r"|missed_|email_domain|email_verification|enrichment|data_vendor"
        r"|country|saved_people)",
        [
            "rox_company",
            "rox_person",
            "rox_lead",
            "person",
            "company",
            "company_request_status",
        ],
    ),
    (
        "lists",
        "Lists, tabs, columns & insights",
        r"^(rox_list|list_runs|tab|column_|custom_column|clever_column|filters"
        r"|folder|rox_default_view|table_metadata|tag$|source$|data_source"
        r"|public_data_source|custom_data_store|csv_upload|contacts_csv|prospect"
        r"|salesforce_import|insight|user_insight|nba_)",
        [
            "rox_list",
            "rox_list_member",
            "column_metadata",
            "data_source",
            "tab",
            "tab_company",
            "insight",
        ],
    ),
    (
        "seq",
        "Sequences & outreach",
        r"^(sequence|campaign|user_template|rox_email|rox_linkedin|tracking_domain"
        r"|custom_email_tracking|outbound_agent|voicemail|call_disposition"
        r"|dial_list|rox_call|twilio|rox_phone|email_ingestion"
        r"|cell_sequence_association$)",
        [
            "campaign_request",
            "sequence",
            "sequence_task",
            "cell_sequence_association",
            "campaign_mailbox_assoc",
            "rox_email",
            "rox_call",
        ],
    ),
    (
        "activity",
        "Activity integrations: email, calendar, meetings, Slack",
        r"^(integration|email_message|calendar|adhoc|unified_|\w+_event_attendee"
        r"|\w+_event_metadata|zoom|ms_teams|transcript|slack|webhook|note$"
        r"|rox_activity|notification|orgwide_integration|call_recorder_|google_drive)",
        [
            "integration",
            "email_message",
            "calendar_event",
            "transcript_v2",
            "slack_channel",
            "webhook_config",
        ],
    ),
    (
        "agents",
        "Agents, chat, workflows & artifacts",
        r"^(conversation|agent_|agentflow|cell_|chat_|skill|prompt|command_config"
        r"|workflow|task_item|domain_agent|app$|app_|shareable|artifact|rox_file"
        r"|user_artifact|user_memory|hotapp$|list_agent_conversation$)",
        [
            "conversation",
            "agent_cell",
            "agent_actions_log",
            "workflow_config",
            "workflow_run",
            "artifact",
            "user_artifact_processing_task_request",
        ],
    ),
]

# Root card layout: one list per column, left to right.
ROOT_COLUMNS: list[list[str]] = [
    ["tenancy", "gov"],
    ["lists", "crm"],
    ["people", "deals"],
    ["seq", "activity"],
    [
        "agents",
        "tasks",
    ],
    [
        "conversation messages",
        "chat:conversation:{conversation_id}:stream:{stream_id}",
    ],
]


def assign(table_names: list[str]) -> dict[str, list[str]]:
    """Return domain id -> sorted member tables; fail on any unassigned table."""
    members: dict[str, list[str]] = {domain_id: [] for domain_id, *_ in DOMAINS}
    unassigned = []
    for name in sorted(table_names):
        if EXCLUDED.match(name):
            continue
        domain_id = next(
            (d for d, _, pattern, _ in DOMAINS if re.search(pattern, name)), None
        )
        if domain_id is None:
            unassigned.append(name)
        else:
            members[domain_id].append(name)
    if unassigned:
        raise SystemExit(f"tables with no domain: {unassigned}")
    for domain_id, _, _, key_tables in DOMAINS:
        missing = [t for t in key_tables if t not in members[domain_id]]
        if missing:
            raise SystemExit(f"key tables not in domain {domain_id}: {missing}")
    return members
