import argparse
import json
import subprocess
from pathlib import Path

import rox_core_domains
from rox_dox.schema import extract_tables

DEFAULT_REPO = Path("/home/ubuntu/repos/rox-core")
argument_parser = argparse.ArgumentParser(
    description="Regenerate the rox-core root page JSON."
)
argument_parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
REPO = argument_parser.parse_args().repo.resolve()
COMMIT = Path(__file__).with_name("COMMIT").read_text(encoding="utf-8").strip()
_cache: dict[str, list[str]] = {}


def lines_of(path: str) -> list[str]:
    if path not in _cache:
        text = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{COMMIT}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        _cache[path] = text.splitlines()
    return _cache[path]


def S(
    path: str,
    needle: str,
    end_needle: str | None = None,
    span: int = 0,
    after: str | None = None,
) -> dict[str, object]:
    lines = lines_of(path)
    start_at = 0
    if after is not None:
        start_at = next(i for i, line in enumerate(lines) if after in line)
    start = next((i for i in range(start_at, len(lines)) if needle in lines[i]), None)
    if start is None:
        raise SystemExit(f"needle not found: {path}: {needle!r}")
    end = start + span
    if end_needle is not None:
        end = next(
            (i for i in range(start, len(lines)) if end_needle in lines[i]), None
        )
        if end is None:
            raise SystemExit(f"end needle not found: {path}: {end_needle!r}")
    return {"path": path, "lines": [start + 1, end + 1]}


def C(text: str, *sources: dict[str, object]) -> dict[str, object]:
    assert 1 <= len(text) <= 90 or True
    return {"text": text, "sources": list(sources)}


def D(text: str, *sources: dict[str, object]) -> dict[str, object]:
    if not 1 <= len(text) <= 90:
        raise SystemExit(f"detail too long ({len(text)}): {text}")
    return C(text, *sources)


EP = "backend/entry-point.sh"
RN = "backend/src/rox_core/api/register_namespaces.py"
LU = "backend/src/util/listener_utils.py"
INIT = "backend/src/rox_core/__init__.py"
BASE = "backend/src/tasks/listeners/base.py"
ROUTER = "backend/src/chat/routes/conversation/router.py"
RUNNER = "backend/src/chat/background_execution/stream_runner.py"
SVC = "backend/src/chat/background_execution/service.py"
RSS = "backend/src/shim_layer/redis_shim/redis_stream_shim.py"
COMPOSE = "backend/docker-compose-infra.yaml"
ASCII = "ascii_architecture.md"

s_testpaths = S("backend/tests_agent/pytest.ini", "testpaths = .")

s_targets_case = S(EP, 'case "$DEPLOY_TARGET" in', "esac")
s_listener_mode = S(
    EP,
    'if [[ "$ASYNC_REQUESTED" == "true" ]]',
    "fi",
    after="SQS Listener Mode Decision",
)
s_gunicorn = S(
    EP,
    'if [[ "$DEPLOY_TARGET" == "REALTIMEAGENT" ]]; then',
    "fi",
    after="Check the DEPLOY_TARGET",
)
s_interaction_gunicorn = S(
    EP,
    'elif [[ "$DEPLOY_TARGET" == "INTERACTION" ]]',
    "run:app --config",
    after="Check the DEPLOY_TARGET",
)
s_keepalive = S(EP, "keep-alive must exceed prod-rox-www-elb")
s_webhook_gunicorn = S(EP, 'elif [[ "$DEPLOY_TARGET" == "WEBHOOK" ]]', span=2)
s_async_agent = S(EP, 'elif [[ "$DEPLOY_TARGET" == "ASYNC_AGENT" ]]', span=1)
s_standalone = S(
    EP,
    'elif [[ "$STANDALONE_REQUESTED" == "true" ]]',
    "exec python run_standalone_consumer.py",
)
s_async_exec = S(
    EP, 'if [[ "$ASYNC_REQUESTED" == "true" ]]', "exec python run_async_listeners.py"
)


RUN_CHAT = "backend/run_chat.py"
RUN_PUB = "backend/run_public_api.py"
RUN_MCP = "backend/run_mcp.py"
RUN_CELL = "backend/run_cell.py"
SCHED = "backend/src/agent_framework/cell_runtime/scheduler.py"
CELLM = "backend/src/agent_framework/cell_runtime/models.py"
JOBM = "backend/src/public_api/async_jobs/jobs/models.py"
JOBS = "backend/src/public_api/async_jobs/jobs/service.py"
DISP = "backend/src/public_api/async_jobs/temporal/dispatch.py"
QABC = "backend/src/shim_layer/queue_shim/queue_abc.py"
KVF = "backend/src/shim_layer/kv/factory.py"
TQ = "backend/src/temporal/queues.py"
TYPES = "backend/src/tasks/types.py"
ROUTING = "backend/src/tasks/routing.py"
NAMES = "backend/src/tasks/names.py"
TWILIO = "backend/src/ext_integrations/twilio_integration/twilio_shim.py"
AUTH0 = "backend/src/shim_layer/auth0_shim/auth0_shim.py"
README = "README.md"
WEB = "web/AGENTS.md"
IOS = "ios/README.md"
QDOC = "docs/queue-abstraction.md"
KVDOC = "docs/kv-store-abstraction.md"
FAPP = "backend/src/chat/server/fastapi_app.py"
MODELS = "backend/src/agent_framework/model"

s_readme = S(README, "# rox-core", "devtool/")
s_rn = S(RN, 'if deploy_target == "INTERACTION":', 'raise ValueError(f"DEPLOY_TARGET')
s_rn_interaction = S(
    RN, 'if deploy_target == "INTERACTION":', 'elif deploy_target == "REALTIMEAGENT":'
)
s_rn_realtime = S(
    RN, 'elif deploy_target == "REALTIMEAGENT":', 'elif deploy_target == "SOR":'
)
s_rn_sor = S(RN, 'elif deploy_target == "SOR":', 'elif deploy_target == "BATCH":')
s_rn_batch = S(
    RN, 'elif deploy_target == "BATCH":', 'elif deploy_target == "EXTRACTIONAGENT":'
)
s_rn_agent = S(RN, 'elif deploy_target == "AGENT":', 'elif deploy_target == "BACKFILL"')
s_rn_fastapi = S(RN, '    deploy_target == "CHAT"', "# Uses FastAPI")
s_init_webhook = S(
    INIT,
    'if deploy_target_for_blueprints == "WEBHOOK":',
    "app.register_blueprint(twilio_webhook_v2_bp)",
)
s_rn_unknown = S(RN, 'raise ValueError(f"DEPLOY_TARGET', span=0)
s_rn_sf = S(RN, "salesforce_enrichment_ns, path=", "hubspot_enrichment_ns, path=")
s_rn_dialer = S(
    RN, 'api.add_namespace(dialer_ns, path="/dialer"', "twilio_webhook_v2_ns, path="
)
s_rn_auth0 = S(RN, 'api.add_namespace(auth0_ns, path="/auth0"')
s_qc = S(LU, "QUEUE_CONFIGS = {", "}")
s_qc_outreach = S(LU, '"WORKFLOWSCHEDULER": (', '"OUTREACH": (')
s_temporal = S(INIT, 'if deploy_target == "TEMPORAL_WORKER":')
s_temporal_enr = S(INIT, 'if deploy_target == "ENRICHMENT_TEMPORAL_WORKER":')
s_temporal_wf = S(INIT, 'if deploy_target == "WORKFLOW_TEMPORAL_WORKER":')
s_leader = S(
    INIT,
    'if not db_mode and deploy_target == "WORKFLOWSCHEDULER":',
    "start_leader_election(app)",
)
s_tq = S(TQ, "class TemporalQueues:", "OUTBOUND_PROSPECTING =")
s_poll = S(BASE, "messages = get_queue_shim().receive_messages(")
s_state_check = S(
    BASE, "current_state = get_current_state(task_json)", "reset_task_state(task_json)"
)
s_exec = S(BASE, "future = execute_task(", "listener_instance=self,")
s_qdoc = S(QDOC, "The Queue abstraction provides", "logic.")
s_kvdoc = S(KVDOC, "The KVStore abstraction provides")
s_kvf = S(KVF, "def _resolve_kv_provider", 'os.getenv("KV_PROVIDER"')
s_qabc = S(QABC, "class QueueShim(ABC):", "def send_message(")
s_router_qr = S(
    ROUTING,
    "QUEUE_ROUTES: dict[",
    "TaskQueueType, QueueCatalogEntry | dict[TaskPriority, QueueCatalogEntry]",
)
s_run_chat = S(RUN_CHAT, "from chat.server.fastapi_app import app", "uvicorn.run(")
s_run_pub = S(RUN_PUB, "from public_api.server.app import app", "uvicorn.run(")
s_run_mcp = S(
    RUN_MCP, "from external_mcp.mcp_server import starlette_app", "uvicorn.run("
)
s_post = S(
    ROUTER,
    '@conversation_stream_router.post("/message/{conversation_id}")',
    '"""Send one message',
)
s_get = S(ROUTER, '"/message/stream/{conversation_id}"', span=1)
s_resume = S(ROUTER, "if last_event_id and not re.match", 'last_event_id = "0-0"')
s_sse = S(ROUTER, "return EventSourceResponse(", span=0)
s_start_turn = S(
    ROUTER, "future = FastAPIEventLoop.submit_coroutine(", "return future.result()"
)
s_rps = S(SVC, "Reserve → persist → spawn", span=3)
s_stream_key = S(RUNNER, "def _get_stream_key", span=1)
s_append = S(RUNNER, "if not await redis_stream.append(event):", span=0)
s_xadd = S(RSS, "async def _xadd(", "await self.redis_client.xadd(")
s_cell_main = S(RUN_CELL, '"""Cell service: boots the core app', span=0)
s_cell_sched = S(RUN_CELL, "scheduler = CellScheduler(loop, app)", "scheduler.start()")
s_sched_cls = S(SCHED, "class CellScheduler:", "One per process.")
s_sched_mongo = S(
    SCHED, "# The runtime collection is one shared Mongo DB", "ROX_SYSTEM_ORG_ID"
)
s_cell_doc = S(CELLM, '"""A cell\'s identity', '__tablename__ = "agent_cell"')
s_job_table = S(
    JOBM,
    "class ApiJob(Base):",
    "unique=True,\n" if False else '"uix_api_jobs_idempotency_key"',
)
s_job_submit = S(JOBS, "async def submit(", "return resource")
s_dispatch = S(
    DISP,
    "handle = await client.start_workflow(",
    "id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE",
)
s_pg = S(COMPOSE, "  postgres:", span=1)
s_redis = S(COMPOSE, "  redis:", span=1)
s_mongo = S(COMPOSE, "  mongo:", span=1)
s_localstack = S(COMPOSE, "  localstack:", span=1)
s_temporal_srv = S(COMPOSE, "  temporal:", span=1)
s_ascii_hl = S(ASCII, "## 1. High-Level System Architecture", "Primary DB")
s_ascii_int = S(ASCII, "## 5. Integration Layer Architecture", "Auth0 Service")
s_web = S(WEB, "This is a pnpm monorepo", span=0)
s_web_q = S(WEB, "react-query", span=0)
s_ios = S(IOS, "SwiftUI iOS app", "extension.")
s_ios_rt = S(IOS, "`RoxIOS/` — app target", "(WebRTC voice)")
s_ios_auth = S(IOS, "Dependencies are Swift packages", "SimpleKeychain")
s_twilio = S(TWILIO, "class TwilioShim:", "self.client = Client(")
s_auth0 = S(AUTH0, "class Auth0Shim:")
s_anthropic = S(f"{MODELS}/rox_anthropic_model.py", "class RoxAnthropicModel(")
s_openai = S(
    f"{MODELS}/rox_openai_responses_model.py", "class RoxOpenAIResponsesModel("
)
s_vertex = S(
    f"{MODELS}/rox_litellm_vertex_ai_model.py", "class RoxLitellmVertexAiModel("
)
s_samba = S(
    f"{MODELS}/rox_litellm_sambanova_model.py", "class RoxLitellmSambanovaModel("
)
s_chat_routers = S(
    FAPP,
    "app.include_router(conversation_stream_router)",
    "app.include_router(conversation_realtime_router)",
)
s_task_table = S("backend/src/rox_core/models/task.py", '__tablename__ = "task_run"')
s_conv_table = S(
    "backend/src/rox_core/models/conversation/conversation.py",
    '__tablename__ = "conversation"',
)
s_integration_table = S(
    "backend/src/rox_core/models/integration.py",
    '__tablename__ = "integration"',
)
s_task_final_state = S(
    "backend/src/tasks/task_handler.py",
    "to_state = TaskState.COMPLETED",
    after="class TaskHandler:",
)
s_task_pg_write = S(
    "backend/src/tasks/executors/calendar_extract_and_persist_events.py",
    "trx_mgr.session.execute(stmt)",
)
s_task_provider_call = S(
    "backend/src/tasks/executors/calendar_extract_and_persist_events.py",
    "GoogleOrgLinkingCalendar.get_or_refresh_admin_creds_if_expired(",
)
s_app_calendar = S(
    "backend/src/rox_core/api/account/service.py",
    "from ext_integrations.calendar_integration.calendar_data_store_v2 import (",
)
s_app_stores = S(
    "backend/src/rox_core/api/account/service.py",
    "from shim_layer.redis_shim.redis import RedisClient",
)
s_google_calendar = S(
    "backend/src/ext_integrations/calendar_integration/google_calendar.py",
    "class GoogleCalendar(BaseGoogleCalendar):",
)
s_gmail_client = S(
    "backend/src/ext_integrations/email_integration/gmail_service.py",
    "from googleapiclient.discovery import build",
)
s_outlook_service = S(
    "backend/src/ext_integrations/email_integration/outlook_service.py",
    "class OutlookService(BaseOutlookService):",
)
s_slack_sdk = S(
    "backend/src/ext_integrations/slack_integration/common.py",
    "from slack_sdk.web import SlackResponse, WebClient",
)
s_microsoft_sdk = S(
    "backend/src/ext_integrations/ms_teams/bot/token_util.py",
    "from msal import ConfidentialClientApplication",
)
s_unipile = S(
    "backend/src/ext_integrations/unipile/unipile_shim.py",
    "class UnipileShim:",
)
s_crm_parser = S(
    "backend/src/ext_integrations/integration/crm/hubspot/parser.py",
    "def parse_hubspot_field",
)
s_hubspot_sdk = S(
    "backend/src/ext_integrations/integration/hubspot/hubspot_shim.py",
    "import hubspot",
)
s_salesforce_sdk = S(
    "backend/src/ext_integrations/integration/salesforce_shim.py",
    "from simple_salesforce import Salesforce, SFType",
)
s_twilio_sdk = S(
    "backend/src/ext_integrations/twilio_integration/twilio_shim.py",
    "from twilio.rest import Client",
)
s_orum_service = S(
    "backend/src/ext_integrations/orum/orum_service.py",
    "class OrumService",
)
s_call_recorder = S(
    "backend/src/ext_integrations/call_recorders/call_recorder_event_service.py",
    "class CallRecorderEventService:",
)
s_apollo_api = S(
    "backend/src/ext_integrations/data_providers/apollo/apollo_shim.py",
    'APOLLO_BASE_URL = "https://api.apollo.io/api/v1"',
)
s_cognism = S(
    "backend/src/ext_integrations/data_providers/cognism/cognism_shim.py",
    "class CognismShim:",
)
s_auth0_class = S(
    "backend/src/shim_layer/auth0_shim/auth0_shim.py",
    "class Auth0Shim:",
)
s_auth0_management = S(
    "backend/src/shim_layer/auth0_shim/auth0_shim.py",
    "response = requests.get(",
    "headers=headers,",
)
s_kms_class = S(
    "backend/src/shim_layer/kms_shim/aws_kms_shim.py",
    "class AwsKmsShim(KmsShim):",
)
s_kms_client = S(
    "backend/src/shim_layer/kms_shim/aws_kms_shim.py",
    "client = boto3.client(",
    '"kms",',
)
s_secrets_factory = S(
    "backend/src/shim_layer/secrets_manager_shim/secrets_manager_shim.py",
    "def _create_client() -> SecretsManagerClient:",
)
s_secrets_client = S(
    "backend/src/shim_layer/secrets_manager_shim/secrets_manager_shim.py",
    "client = boto3.client(",
    '"secretsmanager",',
)
s_dynamo_client = S(
    "backend/src/shim_layer/dynamo_shim/dynamo_shim.py",
    '_dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)',
)
s_s3_client = S(
    "backend/src/shim_layer/s3_shim/s3_shim.py",
    "client = boto3.client(",
    '"s3",',
)
s_redis_sdk = S(
    "backend/src/shim_layer/redis_shim/redis.py",
    "from redis import Redis, RedisCluster",
)
s_kv_store = S(
    "backend/src/shim_layer/kv/kv_store.py",
    "class KvStore(ABC):",
)
s_queue_shim = S(
    "backend/src/shim_layer/queue_shim/queue_abc.py",
    "class QueueShim(ABC):",
)
s_sqs_shim = S(
    "backend/src/shim_layer/sqs_shim/sqs_shim.py",
    "class SqsQueueShim(QueueShim):",
)
s_sqs_client = S(
    "backend/src/shim_layer/sqs_shim/sqs_shim.py",
    "client = boto3.client(",
    '"sqs",',
)
s_stripe_sdk = S(
    "backend/src/shim_layer/stripe_shim/stripe_shim.py",
    "import stripe",
)
s_rillet_api = S(
    "backend/src/shim_layer/rillet_shim/rillet_shim.py",
    'RILLET_API_BASE_URL = os.getenv("RILLET_API_BASE_URL",',
)
s_observability = S(
    "backend/src/shim_layer/observability/__init__.py",
    '"""Unified observability layer for logging, metrics, and tracing.',
)
s_sns_push = S(
    "backend/src/shim_layer/push_notifications_shim/sns_push_shim.py",
    "class SnsPushShim(PushShim):",
)
s_google_workspace_queue_type = S(
    NAMES,
    "GOOGLE_WORKSPACE_ADMIN_CREATE_TASK = TaskCatalogEntry(",
    '"GOOGLE_WORKSPACE_ADMIN_CREATE", IntegrationQueueType.INTEGRATION',
)
s_integrations_route = S(
    "backend/src/rox_core/api/integrations/endpoints.py",
    '@integrations_ns.route("", endpoint="integration_list")',
)
s_create_integration_call = S(
    "backend/src/rox_core/api/integrations/endpoints.py",
    "integration = create_integration(",
)
s_create_integration = S(
    "backend/src/rox_core/api/integrations/business.py",
    "def _create_integration(",
)
s_workspace_admin_cache = S(
    "backend/src/rox_core/api/integrations/business.py",
    "cache.set_org_workspace_admin_enabled(",
    after="IntegrationType.GOOGLE_WORKSPACE_ADMIN.value:",
)
s_workspace_admin_task = S(
    "backend/src/rox_core/api/integrations/business.py",
    "task_json = GoogleWorkspaceAdminCreateTask(",
)
s_workspace_admin_queue_call = S(
    "backend/src/rox_core/api/integrations/business.py",
    "task_json = GoogleWorkspaceAdminCreateTask(",
    "queue_task(",
)
s_integration_row_save = S(
    "backend/src/rox_core/api/integrations/business.py",
    "trx_mgr.session.add(integration)",
    after="def _create_integration(",
)
s_workspace_admin_executor = S(
    "backend/src/tasks/executors/google_workspace_admin_integration.py",
    "class GoogleWorkspaceAdminCreateTaskExecutor",
)
s_workspace_admin_type_check = S(
    "backend/src/tasks/executors/google_workspace_admin_integration.py",
    "if integration.type != IntegrationType.GOOGLE_WORKSPACE_ADMIN.value:",
)
s_google_workspace_user_integrations = S(
    "backend/src/ext_integrations/integration/workspace_integration/google_workspace_fanout.py",
    "def ensure_google_workspace_user_integrations_exist(",
)
s_integration_fanout = S(
    "backend/src/ext_integrations/integration/admin_integration_fanout.py",
    "def fan_out_tasks_for_integrations(",
)
s_initialize_calendar_integrations = S(
    "backend/src/ext_integrations/integration/admin_integration_fanout.py",
    "initialize_calendar_integrations(",
    after="def fan_out_tasks_for_integrations(",
)
s_temporal_calendar_start = S(
    "backend/src/ext_integrations/integration/admin_integration_fanout.py",
    "started_temporal_calendar = try_start_orgwide_calendar_temporal_workflow(",
)
s_temporal_email_start = S(
    "backend/src/ext_integrations/integration/admin_integration_fanout.py",
    "started_temporal_email = try_start_orgwide_email_temporal_workflow(",
)
s_temporal_calendar_workflow = S(
    "backend/src/ext_integrations/integration/admin_integration_fanout.py",
    "def try_start_orgwide_calendar_temporal_workflow(",
)
s_sqs_calendar_tasks = S(
    "backend/src/ext_integrations/integration/admin_integration_fanout.py",
    "def queue_calendar_extraction_tasks_for_integrations(",
)
s_new_integration_fanout_call = S(
    "backend/src/ext_integrations/integration/workspace_integration/google_workspace_fanout.py",
    "fan_out_tasks_for_integrations(",
    after="def fan_out_tasks_for_new_integrations(",
)
s_gcal_callback = S(
    "backend/src/ext_integrations/calendar_integration/google_calendar.py",
    "/api/integrations/gcal/callback",
)
s_calendar_refresh_schedule = S(
    "backend/src/rox_core/api/tasks/services.py",
    "def schedule_calendar_extraction_coordinator_task(",
)
s_task_executor_execute = S(
    "backend/src/tasks/task_handler.py",
    "task_executor_result = task_executor.execute()",
)
s_executor_dispatch = S(
    "backend/src/tasks/executor_registry.py",
    "return TASK_EXECUTORS[task_type.task_type](task)",
)
s_task_queue_task = S(
    "backend/src/tasks/producer.py",
    "def queue_task(",
)
s_task_type_queue = S(
    NAMES,
    "class TaskCatalogEntry(NamedTuple):",
    "queue_type: TaskQueueType",
)
s_task_queue_spec = S(
    "backend/src/tasks/queue_registry.py",
    '"CALENDAR_EXTRACTION_COORDINATOR_TASK": QueueCatalogEntry(',
)
s_task_listener_run = S(
    "backend/src/tasks/listeners/base.py",
    "def run(self) -> None:",
)
s_listener_queue_configs = S(
    "backend/src/util/listener_utils.py",
    "QUEUE_CONFIGS = {",
)
s_listener_receive = S(
    "backend/src/tasks/listeners/base.py",
    "messages = get_queue_shim().receive_messages(",
)
s_listener_drop_states = S(
    "backend/src/tasks/listeners/base.py",
    "TaskState.COMPLETED.value,",
    "TaskState.SKIPPED.value,",
)
s_task_handler_class = S(
    "backend/src/tasks/task_handler.py",
    "class TaskHandler:",
)
s_task_executor_resolution = S(
    "backend/src/tasks/task_handler.py",
    "task_executor = get_executor(self.task_json)",
)
s_executor_registry = S(
    "backend/src/tasks/executor_registry.py",
    "def get_executor(task_json: dict)",
)
s_email_executor = S(
    "backend/src/tasks/executors/email.py",
    "class EmailIntegrationTaskExecutor",
)
s_calendar_executor = S(
    "backend/src/tasks/executors/calendar_extract_and_persist_events.py",
    "class CalendarExtractionCoordinatorTaskExecutor",
)
s_data_extraction_executor = S(
    "backend/src/tasks/executors/data_extraction_coordinator_executor.py",
    "class DataExtractionCoordinatorExecutor",
)
s_chat_mermaid = S(
    "backend/src/chat/background_execution/README.md",
    "```mermaid",
)
s_stream_hook_type = S(
    RUNNER,
    "pre_spawn_hook: Callable[[], Awaitable[None]],",
)
s_chat_post_route = S(
    ROUTER,
    '@conversation_stream_router.post("/message/{conversation_id}")',
)
s_chat_start_turn = S(
    SVC,
    "async def start_turn(",
)
s_chat_stream_initialize = S(
    RUNNER,
    "await redis_stream.initialize_stream(",
)
s_chat_fail_if_active = S(
    RUNNER,
    "fail_if_active=True,",
)
s_chat_persist_hook = S(
    SVC,
    "pre_spawn_hook=persist_user_turn,",
)
s_chat_spawn_task = S(
    RUNNER,
    "task = asyncio.create_task(",
)
s_chat_runner_run = S(
    RUNNER,
    "    async def run(",
)
s_chat_heartbeat = S(
    RUNNER,
    "_send_heartbeat_with_span(",
)
s_chat_append_heartbeat = S(
    RUNNER,
    "_send_heartbeat_with_span(",
    "if not await redis_stream.append(event):",
    after="    async def run(",
)
s_agent_event_generator = S(
    "backend/src/chat/background_execution/business.py",
    "async def agent_event_generator(",
)
s_rox_runner_streaming = S(
    "backend/src/agent_framework/runner/rox_runner.py",
    "    async def _start_streaming(",
)
s_chat_get_route = S(
    ROUTER,
    '"/message/stream/{conversation_id}",',
)
s_chat_redis_read = S(
    "backend/src/chat/background_execution/business.py",
    "redis_stream = RedisStreamShim(stream_key, last_event_id)",
    "async for stream_data in redis_stream.read_all():",
)
s_chat_cfg = S(
    SVC,
    "chat_config, trace_metadata = await TemporaryAgentOffloadPool.run_in_executor(",
)

groups = [
    {"id": "clients", "label": "Clients", "source": s_readme},
    {"id": "http", "label": "HTTP services", "source": s_rn},
    {"id": "queues", "label": "Queues + workflow engine", "source": s_qabc},
    {"id": "workers_col", "label": "Background workers", "source": s_targets_case},
    {
        "id": "workers",
        "label": "SQS listener targets",
        "parent": "workers_col",
        "source": s_qc,
    },
    {
        "id": "durable",
        "label": "Durable / long-running",
        "parent": "workers_col",
        "source": s_temporal,
    },
    {"id": "data", "label": "Stores", "source": s_pg},
    {"id": "external", "label": "External providers", "source": s_ascii_int},
    {
        "id": "providers",
        "label": "Integration providers",
        "parent": "external",
        "source": s_rn_sf,
    },
]


def N(
    id: str,
    label: str,
    group: str,
    source: dict[str, object],
    details: list[dict[str, object]],
    kind: str = "component",
    many: bool = False,
) -> dict[str, object]:
    return {
        "id": id,
        "label": label,
        "group": group,
        "source": source,
        "kind": kind,
        "many": many,
        "details": details,
    }


nodes = [
    N(
        "web",
        "web (Next.js 14)",
        "clients",
        s_web,
        [
            D("pnpm monorepo; app in web/apps/web", s_web),
            D("react-query hooks generated by orval from swagger", s_web_q),
        ],
    ),
    N(
        "ios",
        "ios (SwiftUI)",
        "clients",
        s_ios,
        [
            D("RoxIOS app + RoxVoiceWidget Live Activity", s_ios),
            D("WebRTC realtime voice; generated API client", s_ios_rt),
            D("Auth0 login via Swift package", s_ios_auth),
        ],
    ),
    N(
        "interaction",
        "INTERACTION (Flask)",
        "http",
        s_rn_interaction,
        [
            D(
                "Main REST API: /user, /integrations, /sequences, /dialer, /billing…",
                s_rn_interaction,
            ),
            D("Gunicorn 2 workers × 50 threads, timeout 123s", s_interaction_gunicorn),
            D("Keep-alive 200s outlasts the ALB 180s idle timeout", s_keepalive),
            D("Also consumes InteractionQueueType SQS queues", s_qc),
        ],
        many=True,
    ),
    N(
        "chat",
        "CHAT (FastAPI)",
        "http",
        s_run_chat,
        [
            D("uvicorn chat.server.fastapi_app", s_run_chat),
            D("POST /message/{id} starts a background turn", s_post),
            D("GET /message/stream/{id} serves SSE, resumable", s_get, s_resume),
            D("Sibling uvicorn apps: FASTAPI_INTERACTION, EVAL", s_targets_case),
        ],
    ),
    N(
        "public_api",
        "PUBLIC_API (FastAPI)",
        "http",
        s_run_pub,
        [
            D("uvicorn public_api.server.app", s_run_pub),
            D("Async jobs accepted into Postgres api_jobs", s_job_submit),
            D("Each job dispatched as a Temporal workflow", s_dispatch),
        ],
    ),
    N(
        "webhook",
        "WEBHOOK (Flask)",
        "http",
        s_init_webhook,
        [
            D("Provider webhook blueprints only, no api_bp", s_init_webhook),
            D("Gunicorn 2 × 50 threads, no worker timeout", s_webhook_gunicorn),
        ],
        many=True,
    ),
    N(
        "mcp",
        "MCP server",
        "http",
        s_run_mcp,
        [
            D("uvicorn external_mcp.mcp_server Starlette app", s_run_mcp),
        ],
    ),
    N(
        "agent_workers",
        "AGENT · REALTIMEAGENT · EXTRACTIONAGENT · ASYNC_AGENT",
        "workers",
        s_qc,
        [
            D("AGENT: /tasks, /insights, /digests, /graph", s_rn_agent),
            D("REALTIMEAGENT: research and enrichment routes", s_rn_realtime),
            D("REALTIMEAGENT/OUTREACH can run a standalone consumer", s_standalone),
            D(
                "ASYNC_AGENT runs only async listeners, no Gunicorn",
                s_async_agent,
                s_async_exec,
            ),
        ],
        many=True,
    ),
    N(
        "data_workers",
        "SOR · BATCH · BACKFILL · INTEGRATION",
        "workers",
        s_qc,
        [
            D("SOR: graph + system-of-record task routes", s_rn_sor),
            D("BATCH: batch task routes", s_rn_batch),
            D("Each target polls its own QueueType enum", s_qc),
        ],
        many=True,
    ),
    N(
        "outreach_sched",
        "OUTREACH · WORKFLOWSCHEDULER",
        "workers",
        s_qc_outreach,
        [
            D("OutreachQueueType / WorkflowSchedulerQueueType", s_qc_outreach),
            D("WORKFLOWSCHEDULER runs leader election", s_leader),
        ],
        many=True,
    ),
    N(
        "temporal_workers",
        "Temporal workers (3 targets)",
        "durable",
        s_temporal,
        [
            D(
                "TEMPORAL_WORKER, ENRICHMENT_…, WORKFLOW_TEMPORAL_WORKER",
                s_temporal,
                s_temporal_enr,
                s_temporal_wf,
            ),
            D("Task queues: rox-flow, enrichment, outbound-prospecting…", s_tq),
        ],
        many=True,
    ),
    N(
        "cell",
        "CELL (agent cell runtime)",
        "durable",
        s_cell_main,
        [
            D("No queue listeners; runs CellScheduler", s_cell_main, s_cell_sched),
            D("One scheduler per process; tick = one scheduling round", s_sched_cls),
            D(
                "Cell lease + runtime rows in Mongo agent_cell_runtime",
                s_sched_mongo,
                s_cell_doc,
            ),
        ],
    ),
    N(
        "postgres",
        "PostgreSQL + pgvector",
        "data",
        s_pg,
        [
            D("Primary DB behind SQLAlchemy models", s_ascii_hl),
            D(
                "Holds task_run, conversation, api_jobs, agent_cell…",
                s_task_table,
                s_conv_table,
                s_job_table,
                s_cell_doc,
            ),
        ],
        kind="store",
    ),
    N(
        "redis",
        "Redis",
        "data",
        s_redis,
        [
            D("Chat turn stream chat:conversation:{id}:stream:{sid}", s_stream_key),
            D("XADD message / heartbeat events", s_xadd),
        ],
        kind="store",
    ),
    N(
        "sqs",
        "SQS queues (QueueShim)",
        "queues",
        s_qabc,
        [
            D("App code never calls boto3 SQS directly", s_qdoc),
            D("QUEUE_ROUTES fans one queue type out by priority", s_router_qr),
            D("Local stand-in: LocalStack", s_localstack),
        ],
        kind="queue",
        many=True,
    ),
    N(
        "kv",
        "KV store (DynamoDB or Mongo)",
        "data",
        s_kvf,
        [
            D("KV_PROVIDER selects provider; default dynamodb", s_kvf),
            D("Provider-agnostic interface for DynamoDB + MongoDB", s_kvdoc),
            D("Mongo also runs locally in docker-compose", s_mongo),
        ],
        kind="store",
    ),
    N(
        "temporal_srv",
        "Temporal server",
        "queues",
        s_temporal_srv,
        [
            D("Durable workflow engine for jobs and flows", s_temporal_srv, s_dispatch),
        ],
        kind="queue",
    ),
    N(
        "llm",
        "LLM providers",
        "external",
        s_anthropic,
        [
            D("Anthropic", s_anthropic),
            D("OpenAI Responses", s_openai),
            D("Vertex AI and SambaNova via LiteLLM", s_vertex, s_samba),
        ],
        kind="external",
    ),
    N(
        "crm",
        "CRM + workspace APIs",
        "providers",
        s_rn_sf,
        [
            D("Salesforce and HubSpot enrichment routes", s_rn_sf),
            D("Google, Slack, Microsoft APIs", s_ascii_int),
            D("INTERACTION calls: enrichment", s_rn_sf),
        ],
        kind="external",
    ),
    N(
        "twilio",
        "Twilio",
        "providers",
        s_twilio,
        [
            D("TwilioShim wraps twilio Client per org", s_twilio),
            D("/dialer and /twilio_webhook_v2 routes", s_rn_dialer),
            D("INTERACTION calls: dialer + webhooks", s_rn_dialer),
        ],
        kind="external",
    ),
    N(
        "auth0",
        "Auth0",
        "providers",
        s_auth0,
        [
            D("Auth0Shim; /auth0 namespace", s_auth0, s_rn_auth0),
            D("INTERACTION calls: /auth0", s_rn_auth0),
        ],
        kind="external",
    ),
]


def E(
    src: str,
    dst: str,
    label: str,
    source: dict[str, object],
) -> dict[str, object]:
    return {"src": src, "dst": dst, "label": label, "source": source}


edges = [
    E("web", "interaction", "REST (orval hooks)", s_web_q),
    E("web", "chat", "POST message, then SSE", s_start_turn),
    E("interaction", "sqs", "enqueue task", s_ascii_hl),
    E("sqs", "agent_workers", "long-poll, execute_task", s_poll),
    E("sqs", "data_workers", "per-target queue types", s_qc),
    E("sqs", "outreach_sched", "per-target queue types", s_qc_outreach),
    E("agent_workers", "postgres", "check task_run state", s_state_check),
    E("chat", "redis", "append turn events", s_append),
    E("chat", "llm", "turn model calls", s_chat_cfg),
    E("public_api", "postgres", "accept api_jobs row", s_job_submit),
    E("public_api", "temporal_srv", "start_workflow", s_dispatch),
    E("temporal_srv", "temporal_workers", "task queues", s_tq),
    E("cell", "kv", "lease runtime rows", s_sched_mongo),
    E("interaction", "providers", "provider calls", s_rn_sf),
]

tldr = {
    "summary": [
        C(
            "rox-core is a monorepo: Python backend, Next.js web, SwiftUI iOS, and the rox-cloud CI CLI.",
            s_readme,
        ),
        C(
            "Every backend service runs the same image; DEPLOY_TARGET picks a FastAPI server, Flask on Gunicorn, an SQS consumer, Temporal workers, or the agent cell scheduler.",
            s_targets_case,
            s_listener_mode,
            s_gunicorn,
        ),
    ],
    "key_points": [
        C(
            "Flask route registration covers 20 deploy targets; any other value raises ValueError.",
            s_rn,
            s_rn_unknown,
        ),
        C(
            "11 targets consume SQS; each maps to one QueueType enum in tasks.types.",
            s_qc,
        ),
        C(
            "The SQS listener checks task_run state first: finished messages are deleted, stuck or failed tasks are reset to QUEUED.",
            s_state_check,
        ),
        C(
            "Chat turns are decoupled from HTTP: reserve stream, persist message, spawn producer; the reply is a resumable SSE read from Redis.",
            s_rps,
            s_resume,
        ),
        C(
            "Public API jobs: the Postgres api_jobs row is accepted first, then a Temporal workflow is started with a deterministic, duplicate-rejecting id.",
            s_job_submit,
            s_dispatch,
        ),
        C(
            "Queues and KV go through shim_layer; KV defaults to DynamoDB with MongoDB as the alternative.",
            s_qdoc,
            s_kvf,
        ),
    ],
    "table": {
        "columns": ["DEPLOY_TARGET", "Process", "Role"],
        "rows": [
            [
                "INTERACTION",
                "Gunicorn (Flask), 2×50, timeout 123s",
                "Main REST API + interaction queues",
            ],
            [
                "CHAT · FASTAPI_INTERACTION · EVAL",
                "uvicorn",
                "Chat turns + SSE; FastAPI REST; evals",
            ],
            ["PUBLIC_API", "uvicorn", "External API, async jobs"],
            ["MCP", "uvicorn (Starlette)", "MCP server"],
            [
                "WEBHOOK",
                "Gunicorn, timeout 0",
                "Webhooks: Circleback, Teams, Twilio, Auth0, Unipile, tracking",
            ],
            [
                "AGENT · REALTIMEAGENT · EXTRACTIONAGENT · SOR · BATCH · BACKFILL · INTEGRATION · OUTREACH · WORKFLOWSCHEDULER",
                "Gunicorn + in-process SQS listeners",
                "Queue workers",
            ],
            ["ASYNC_AGENT", "run_async_listeners.py", "Async SQS consumer only"],
            [
                "TEMPORAL_WORKER · ENRICHMENT_TEMPORAL_WORKER · WORKFLOW_TEMPORAL_WORKER",
                "Gunicorn + Temporal worker",
                "Durable workflows",
            ],
            ["CELL", "run_cell.py", "CellScheduler for agent cells"],
        ],
        "sources": [
            s_targets_case,
            s_listener_mode,
            s_gunicorn,
            s_rn,
            s_init_webhook,
            s_qc,
            s_temporal,
            s_temporal_enr,
            s_temporal_wf,
            s_cell_main,
        ],
    },
    "notes": [
        C(
            "devtool/ is marked legacy in the root README.",
            S(README, "[`devtool/`](devtool/)", "nothing in the docs above uses it"),
        ),
        C(
            "Locally, docker-compose stands in Postgres, Redis, Mongo, LocalStack and Temporal for AWS and cloud services.",
            s_pg,
            s_localstack,
            s_temporal_srv,
        ),
        C(
            "Tests and migrations are not documented on this site.",
            s_testpaths,
        ),
    ],
}

BUS = "backend/src/tasks/run_queries.py"
PRODUCER = "backend/src/tasks/producer.py"
RUNSVC = "backend/src/tasks/run_service.py"
TH = "backend/src/tasks/task_handler.py"
TASKM = "backend/src/rox_core/models/task.py"
CONVM = "backend/src/rox_core/models/conversation/conversation.py"
CSVC = "backend/src/chat/routes/conversation/service.py"
CBUS = "backend/src/chat/background_execution/business.py"
CEXEC = "backend/src/chat/agent_executor/chat_agent_executor.py"
LIFE = "backend/src/public_api/async_jobs/jobs/lifecycle.py"
JDTO = "backend/src/public_api/async_jobs/jobs/dto.py"
WF = "backend/src/public_api/async_jobs/temporal/workflow.py"
ACT = "backend/src/public_api/async_jobs/temporal/activities.py"
ROUTE = "backend/src/public_api/async_jobs/http/route.py"
BGREADME = "backend/src/chat/background_execution/README.md"

# --- background task over SQS -------------------------------------------
s_queue_task = S(
    PRODUCER,
    "def queue_task(",
    ") -> TaskRunDto | UserTaskRunResponseDto:",
)
s_create_queued = S(
    RUNSVC,
    "current_state = TaskState.QUEUED",
    after="def create_task_run_with_session(",
)
s_send = S(PRODUCER, "get_queue_shim().send_message(queue_url, json.dumps(task_json))")
s_running = S(TH, "to_state=TaskState.RUNNING,", 'message="Task is running",')
s_execute = S(TH, 'results["execute"] = self.get_traced_execute(task_executor)()')
s_completed = S(TH, "to_state = TaskState.COMPLETED", 'message = "Task is completed"')
s_setup_failed = S(
    TH,
    "to_state=TaskState.FAILED,",
    span=1,
    after="task_executor.init(self.rox_org_id, self.task_run_id)",
)
s_th_failed = S(
    TH, "to_state = TaskState.FAILED", span=1, after="except DeferredTaskTransition:"
)
s_th_stopped = S(TH, "if skip_execution:", "to_state = TaskState.STOPPED")
s_th_skipped = S(
    TH, "except SkippedTaskTransition as e:", "to_state = TaskState.SKIPPED"
)
s_finish = S(TH, 'results["finish"] = task_executor.finish(to_state)')
s_delete_ok = S(
    BASE,
    "get_queue_shim().delete_message(queue_to_poll, receipt_handle)",
    "# Increment completion counter (successful task completion)",
    after="# Increment completion counter (task processed, even though it failed)",
)
s_finished_skip = S(
    BASE,
    "if current_state in [",
    "get_queue_shim().delete_message(",
    after="current_state = get_current_state(task_json)",
)
s_reset = S(
    BASE,
    "if current_state in [",
    "reset_task_state(task_json)",
    after="self.tasks_completed += 1",
)
s_vst = S(
    BUS,
    "VALID_STATE_TRANSITIONS = {",
    "TaskState.SKIPPED.value: [TaskState.SKIPPED.value],",
)
s_vst_created = S(
    BUS, "TaskState.CREATED.value: [", "],", after="VALID_STATE_TRANSITIONS = {"
)
s_vst_queued = S(
    BUS, "TaskState.QUEUED.value: [", "],", after="VALID_STATE_TRANSITIONS = {"
)
s_vst_running = S(
    BUS, "TaskState.RUNNING.value: [", "],", after="VALID_STATE_TRANSITIONS = {"
)
s_vst_completed = S(BUS, "TaskState.COMPLETED.value: [TaskState.STOPPED.value],")
s_vst_failed = S(
    BUS, "TaskState.FAILED.value: [TaskState.QUEUED.value, TaskState.STOPPED.value],"
)


def TS(name: str) -> dict:
    return S(TASKM, f'{name} = "{name}"', after="class TaskState(Enum):")


# --- chat turn ----------------------------------------------------------
s_runner_cls = S(RUNNER, "class ConversationStreamRunner", span=0)
s_init_stream = S(RUNNER, "await redis_stream.initialize_stream(tags=self._dd_tags())")
s_reserve = S(
    RUNNER,
    "conv_service.manage_stream_id,",
    "fail_if_active=True,",
    after="async def start(",
)
s_hook = S(RUNNER, "await pre_spawn_hook()")
s_persist = S(
    SVC,
    "async def persist_user_turn() -> None:",
    "attached_file_ids=attached_file_ids,",
)
s_spawn = S(RUNNER, "task = asyncio.create_task(", "return stream_id")
s_gen = S(RUNNER, "async for event in gen:", span=1)
s_agent_gen = S(CBUS, "async def agent_event_generator(", span=0)
s_chat_exec = S(CEXEC, "class ChatAgentExecutor(", span=0)
s_expire_call = S(RUNNER, "await redis_stream.expire_stream()")
s_cleanup_call = S(
    RUNNER, "conv_service.cleanup_stream_id,", span=2, after="if cleanup_stream:"
)
s_read = S(
    CBUS,
    "redis_stream = RedisStreamShim(stream_key, last_event_id)",
    "async for stream_data in redis_stream.read_all():",
)
s_conv_col = S(CONVM, "redis_stream_id: Mapped[str | None]")
s_msi = S(
    CSVC,
    "if fail_if_active and current_stream_id:",
    "conversation.redis_stream_id = new_stream_id",
)
s_csi = S(
    CSVC,
    "if not current_stream_id or current_stream_id != expected_stream_id:",
    "conversation.redis_stream_id = None",
)
s_create_msgs = S(
    CSVC,
    "def create_messages(",
    "The DynamoDB write follows only after Postgres succeeds.",
)
s_sort_key = S(CSVC, "Rows in one batch get `created_on = base_ms + index`", span=1)
s_bg_readme = S(BGREADME, "# Background Chat Streaming", span=0)
s_lock_doc = S(
    BGREADME,
    "The conversation row's `redis_stream_id` column is both a **pointer**",
    "means the message is durable",
)
s_sdt = S(RSS, 'MESSAGE = "message"', 'TIMED_OUT = "timed_out"')
s_rs_init = S(RSS, '{"type": StreamDataType.INITIALIZED.value}')
s_rs_append = S(RSS, '{"type": StreamDataType.MESSAGE.value, "event": event},')
s_rs_heartbeat = S(RSS, '{"type": StreamDataType.HEARTBEAT.value}')
s_rs_expire = S(
    RSS, "async def expire_stream(self) -> None:", "await self._expire_stream()"
)
s_rs_cancel = S(
    RSS, "async def cancel_stream(self) -> None:", "await self._expire_stream()"
)
s_rs_timeout = S(
    RSS, "await stream._set_canceled()", '{"type": StreamDataType.TIMED_OUT.value}'
)
s_start_fail = S(
    RUNNER,
    "except Exception:",
    "await redis_stream.cancel_stream()",
    after="await pre_spawn_hook()",
)

# --- public API async job ----------------------------------------------
s_route = S(ROUTE, "async def handle(request: Request) -> Response:", span=0)
s_accept = S(
    JOBS,
    "job = self._acceptance.accept(request=validated, idempotency_key=idempotency_key)",
)
s_bg = S(JOBS, "if resource.status == ApiJobStatus.QUEUED:", span=1)
s_candidate = S(
    DISP, "def _read_dispatch_candidate", "ApiJob.status == ApiJobStatus.QUEUED,"
)
s_exec_act = S(
    WF,
    "uri: str | None = await self.execute_activity(",
    "retry_policy=params.execution_policy.retry_policy,",
)
s_activities = S(ACT, "class AsyncJobActivities:", span=1)
s_claim = S(
    ACT,
    "job = AsyncJobLifecycle(rox_org_id=params.rox_org_id).claim_for_execution(",
    "if job is None:",
)
s_claim_life = S(
    LIFE, "self._record_transition(job_id, ApiJobStatus.PROCESSING, job is not None)"
)
s_handler = S(ACT, "result = AgentEventLoopPool.submit_coroutine(", ").result()")
s_result_put = S(
    ACT,
    "return AsyncJobResultStore(rox_org_id=job.rox_org_id, job_id=job.id).put(",
    span=2,
)
s_record_success = S(
    WF, "async def _record_success(", "retry_policy=_DB_FINALIZATION_RETRY_POLICY,"
)
s_record_failure = S(
    WF,
    "# Persistence trouble must never send control back to business execution.",
    span=1,
)
s_finalize = S(
    LIFE,
    "status = ApiJobStatus.FAILED if failure is not None else ApiJobStatus.COMPLETED",
)
s_unclaimed_fail = S(LIFE, "Failure allows unclaimed jobs", span=2)
s_job_get = S(JOBS, "async def get(self, *, public_id: UUID)", span=2)
s_job_status = S(JDTO, "class ApiJobStatus(StrEnum):", 'FAILED = "failed"')


def JS(name: str) -> dict:
    return S(JDTO, f"{name} = ", after="class ApiJobStatus(StrEnum):")


def P(pid: str, label: str, source: dict) -> dict:
    return {"id": pid, "label": label, "source": source}


def M(src: str, dst: str, message: str, source: dict) -> dict:
    return {"src": src, "dst": dst, "message": message, "source": source}


sequences = [
    {
        "title": "Background task over SQS",
        "participants": [
            P("caller", "Any service (queue_task)", s_queue_task),
            P("pg", "Postgres task_run", s_task_table),
            P("sqs", "SQS (QueueShim)", s_qabc),
            P("listener", "SQS listener", s_poll),
            P("handler", "TaskHandler + executor", s_execute),
        ],
        "steps": [
            M("caller", "pg", "insert task_run (QUEUED)", s_create_queued),
            M("caller", "sqs", "send_message(task_json)", s_send),
            M("listener", "sqs", "receive_messages (long poll)", s_poll),
            M("listener", "pg", "get_current_state(task_run_id)", s_state_check),
            M("listener", "handler", "execute_task on listener executor", s_exec),
            M("handler", "pg", "task_run -> RUNNING", s_running),
            M("handler", "handler", "executor.execute()", s_execute),
            M("handler", "pg", "finish: task_run -> COMPLETED", s_completed),
            M("listener", "sqs", "delete_message(receipt_handle)", s_delete_ok),
        ],
    },
    {
        "title": "Chat turn with a resumable stream",
        "participants": [
            P("web", "Web client", s_web),
            P("chat", "CHAT (FastAPI)", s_post),
            P("runner", "ConversationStreamRunner", s_runner_cls),
            P("pg", "Postgres conversation + DynamoDB messages", s_create_msgs),
            P("redis", "Redis stream", s_stream_key),
            P("agent", "ChatAgentExecutor + LLM", s_chat_exec),
        ],
        "steps": [
            M("web", "chat", "POST /message/{conversation_id}", s_post),
            M("chat", "runner", "start_turn: reserve, persist, spawn", s_rps),
            M("runner", "redis", "initialize_stream (INITIALIZED)", s_init_stream),
            M(
                "runner",
                "pg",
                "reserve redis_stream_id (400 if a turn is active)",
                s_reserve,
            ),
            M("runner", "pg", "pre_spawn_hook: persist user message", s_persist),
            M("runner", "runner", "spawn background producer task", s_spawn),
            M("chat", "web", "200: message is durable", s_hook),
            M("web", "chat", "GET /message/stream/{id} with Last-Event-ID", s_get),
            M("runner", "agent", "agent_event_generator: turn loop", s_agent_gen),
            M("runner", "redis", "append each event (XADD)", s_gen),
            M("chat", "redis", "read_all from last_event_id", s_read),
            M("chat", "web", "SSE events", s_sse),
            M("runner", "redis", "expire_stream (COMPLETED + TTL)", s_expire_call),
            M("runner", "pg", "cleanup_stream_id (release lock)", s_cleanup_call),
        ],
    },
    {
        "title": "Public API async job on Temporal",
        "participants": [
            P("client", "API client", s_route),
            P("pub", "PUBLIC_API (FastAPI)", s_run_pub),
            P("pg", "Postgres api_jobs", s_job_table),
            P("temporal", "Temporal server", s_temporal_srv),
            P("worker", "Temporal worker (AsyncJobActivities)", s_activities),
            P("store", "Result store", s_result_put),
        ],
        "steps": [
            M("client", "pub", "POST job request", s_route),
            M("pub", "pg", "accept: insert api_jobs (queued)", s_accept),
            M("pub", "client", "job resource; dispatch runs after response", s_bg),
            M("pub", "pg", "re-read queued candidate", s_candidate),
            M(
                "pub",
                "temporal",
                "start_workflow (deterministic id, reject duplicates)",
                s_dispatch,
            ),
            M("temporal", "worker", "execute_public_api_job activity", s_exec_act),
            M("worker", "pg", "claim: status -> processing, temporal_run_id", s_claim),
            M("worker", "worker", "operation.execute(request)", s_handler),
            M("worker", "store", "put result -> URI", s_result_put),
            M("temporal", "worker", "finalize_public_api_job(uri)", s_record_success),
            M("worker", "pg", "status -> completed, outcome_ref", s_finalize),
            M("client", "pub", "GET job (poll)", s_job_get),
        ],
    },
]


DEAL = "backend/src/rox_core/models/entities/deal.py"
COLMETA = "backend/src/rox_core/models/column_metadata.py"
SEQ = "backend/src/rox_core/models/sequence.py"
COST = "backend/src/rox_core/models/cost_meter.py"


def T(table: str) -> str:
    return f'__tablename__ = "{table}"'


def R(
    src: str, dst: str, label: str, kind: str, source: dict[str, object]
) -> dict[str, object]:
    return {"src": src, "dst": dst, "label": label, "kind": kind, "source": source}


data = {
    "sql_tables": [],
    "nosql": [
        {
            "name": "chat:conversation:{conversation_id}:stream:{stream_id}",
            "kind": "Redis stream",
            "fields": [
                "type: initialized | message | heartbeat | completed | canceled | timed_out",
                "event: serialized chat event",
            ],
            "source": s_sdt,
        },
        {
            "name": "conversation messages",
            "kind": "DynamoDB (KV store)",
            "fields": [
                "conversation_id",
                "sort key {created_on_ms}#{message_id}",
                "rox_user_id",
                "message data",
            ],
            "source": s_sort_key,
        },
    ],
    "relations": [
        R(
            "deal.rox_company_id",
            "rox_company.rox_company_id",
            "declared FK",
            "enforced",
            S(DEAL, 'ForeignKey("rox_company.rox_company_id")', after=T("deal")),
        ),
        R(
            "deal.sub_org_id",
            "sub_organization.public_id",
            "declared FK",
            "enforced",
            S(DEAL, 'ForeignKey("sub_organization.public_id"', after=T("deal")),
        ),
        R(
            "entity_company.rox_id",
            "rox_company.rox_company_id",
            "query join: CRM account to Rox company",
            "symbolic",
            S(
                "backend/src/ext_integrations/integration/crm/salesforce/"
                "writeback_engine_query_generator.py",
                ".outerjoin(Account, Account.rox_id == RoxCompany.rox_company_id)",
            ),
        ),
        R(
            "external_column_mapping.column_id",
            "column_metadata.public_id",
            "ORM primaryjoin",
            "symbolic",
            S(
                COLMETA,
                'primaryjoin="ColumnMetadata.public_id=='
                'foreign(ExternalColumnMapping.column_id)"',
            ),
        ),
        R(
            "crm_field.data_source_public_id",
            "data_source.public_id",
            "declared FK",
            "enforced",
            S(
                "backend/src/rox_core/models/crm_field.py",
                'ForeignKey("data_source.public_id")',
            ),
        ),
        R(
            "profile_permission.profile_id",
            "user_profile.profile_id",
            "declared FK",
            "enforced",
            S(
                "backend/src/rox_core/models/multiplayer/user_profile.py",
                'ForeignKey("user_profile.profile_id")',
                after=T("profile_permission"),
            ),
        ),
        R(
            "tab_company.rox_company_id",
            "rox_company.rox_company_id",
            "declared FK",
            "enforced",
            S(
                "backend/src/rox_core/models/tab.py",
                'ForeignKey("rox_company.rox_company_id")',
                after=T("tab_company"),
            ),
        ),
        R(
            "rox_list_member.entity_id",
            "rox_lead.rox_lead_id",
            "query join: list member is a lead",
            "symbolic",
            S(
                "backend/src/rox_core/api/prospect_lists/service.py",
                "RoxListMember.entity_id == model.rox_lead_id,",
            ),
        ),
        R(
            "sequence.rox_person_id",
            "rox_person.rox_person_id",
            "symbolic FK (model comment)",
            "symbolic",
            S(
                SEQ,
                "rox_person_id: Mapped[str | None] = mapped_column(",
                span=2,
                after=T("sequence"),
            ),
        ),
        R(
            "sequence.rox_lead_id",
            "rox_lead.rox_lead_id",
            "symbolic FK (model comment)",
            "symbolic",
            S(
                SEQ,
                "# Symbolic fk w/ rox_lead table.",
                end_needle="rox_lead_id: Mapped",
            ),
        ),
        R(
            "cell_sequence_association.cell_id",
            "agent_cell.cell_id",
            "one regen cell per sequence",
            "symbolic",
            S(
                "backend/src/outreach/seqregen_cell/models.py",
                '"""One row per sequence with a regen cell',
                end_needle="cell_id: Mapped[str] = mapped_column(",
            ),
        ),
        R(
            "campaign_mailbox_assoc.integration_id",
            "integration.public_id",
            "declared FK",
            "enforced",
            S(
                "backend/src/rox_core/models/campaign_request.py",
                'ForeignKey("integration.public_id"',
                after=T("campaign_mailbox_assoc"),
            ),
        ),
        R(
            "rox_email.message_id",
            "email_message.message_id",
            "ORM primaryjoin",
            "symbolic",
            S(
                "backend/src/rox_core/models/rox_email.py",
                "primaryjoin=lambda: and_(",
                end_needle="RoxEmail.creator_email_address == foreign(PublicEmailMessage.email),",
                after="public_email_message: Mapped",
            ),
        ),
        R(
            "integration.created_by",
            "user.rox_user_id",
            "declared FK",
            "enforced",
            S(
                "backend/src/rox_core/models/integration.py",
                "created_by: Mapped[str] = mapped_column(",
                span=1,
                after=T("integration"),
            ),
        ),
        R(
            "agent_actions_log.rox_company_id",
            "rox_company.rox_company_id",
            "commented-out FK",
            "symbolic",
            S(
                COST,
                '# ForeignKey("rox_company.rox_company_id")',
                after=T("agent_actions_log"),
            ),
        ),
        R(
            "agent_actions_log.run_id",
            "task_run.run_id",
            "commented-out FK",
            "symbolic",
            S(COST, '# ForeignKey("task_run.run_id")', after=T("agent_actions_log")),
        ),
        R(
            "user_artifact_processing_task_request.extraction_task_id",
            "extraction_task_status.extraction_task_id",
            "declared FK",
            "enforced",
            S(
                "backend/src/rox_core/models/user_artifact_processing_task_request.py",
                'ForeignKey("extraction_task_status.extraction_task_id")',
            ),
        ),
        R(
            "user_task_run.rox_company_id",
            "rox_company.rox_company_id",
            "commented-out FK",
            "symbolic",
            S(
                "backend/src/rox_core/models/task.py",
                '# ForeignKey("rox_company.rox_company_id")',
                after=T("user_task_run"),
            ),
        ),
        R(
            "cost_meter_log.run_id",
            "workflow_run.temporal_workflow_run_id",
            "run cost summed per workflow run",
            "symbolic",
            S(
                "backend/src/workflow/workflow_run/workflow_run_service.py",
                "CostMeterLog.run_id == WorkflowRun.temporal_workflow_run_id,",
            ),
        ),
        R(
            "company_request_status.parent_task_run_id",
            "task_run.run_id",
            "commented-out FK",
            "symbolic",
            S(
                "backend/src/rox_core/models/data_extraction_company_request_status.py",
                '# ForeignKey("task_run.run_id")',
                after="parent_task_run_id",
            ),
        ),
        R(
            "conversation messages::conversation_id",
            "conversation.public_id",
            "lookup by public_id",
            "symbolic",
            S(
                "backend/src/chat/routes/conversation/service.py",
                "conversation = Conversation.find_by_public_id(",
                span=1,
                after="def get_tag_mapping(",
            ),
        ),
        R(
            "chat:conversation:{conversation_id}:stream:{stream_id}",
            "conversation.public_id",
            "key embeds conversation id",
            "symbolic",
            S(
                "backend/src/chat/background_execution/stream_runner.py",
                "def _get_stream_key(conversation_id: str, stream_id: str) -> str:",
                span=1,
            ),
        ),
        R(
            "chat:conversation:{conversation_id}:stream:{stream_id}",
            "conversation.redis_stream_id",
            "stream id reserved on conversation",
            "symbolic",
            s_reserve,
        ),
    ],
}

tables = extract_tables(REPO, COMMIT)
domain_tables = rox_core_domains.assign(list(tables))
organization_model = "backend/src/rox_core/models/organization.py"
subscription_model = "backend/src/rox_core/models/subscription.py"
tenancy_fk_count = sum(
    column.foreign_key
    in {
        "organization.rox_org_id",
        "user.rox_user_id",
    }
    for table in tables.values()
    for column in table.columns
)
domains = []
for domain_id, title, _, key_tables in rox_core_domains.DOMAINS:
    domain = {
        "id": domain_id,
        "title": title,
        "tables": domain_tables[domain_id],
        "key_tables": key_tables,
        "page": f"domain-{domain_id}",
    }
    if domain_id == "tenancy":
        domain["notes"] = [
            C(
                "rox_org_id → organization and rox_user_id → user on most tables "
                f"({tenancy_fk_count} declared FK columns)",
                S(
                    organization_model,
                    "rox_org_id: Mapped[str] = mapped_column(",
                    after=T("organization"),
                ),
                S(
                    subscription_model,
                    'ForeignKey("organization.rox_org_id")',
                    after=T("subscription"),
                ),
            )
        ]
    else:
        domain["notes"] = []
    domains.append(domain)
data["domains"] = domains
data["columns"] = rox_core_domains.ROOT_COLUMNS


related = [
    {
        "label": "Background chat streaming (README)",
        "url": f"https://github.com/Rox-AI/rox-core/blob/{COMMIT}/{BGREADME}",
        "source": s_bg_readme,
    },
    {
        "label": "ASCII architecture overview",
        "url": f"https://github.com/Rox-AI/rox-core/blob/{COMMIT}/{ASCII}",
        "source": s_ascii_hl,
    },
]

block_figures = [
    {
        "id": "task-pipeline",
        "title": "Background task pipeline",
        "notes": [
            C(
                "Based on ascii_architecture.md §4 (Task Executor Pattern), corrected",
                S(ASCII, "## 4. Task Executor Pattern"),
            ),
            C(
                "Doc's CalendarSyncTaskExecutor / DataExtractionTaskExecutor don't "
                "exist; current names shown",
                s_calendar_executor,
                s_data_extraction_executor,
            ),
        ],
        "block": {
            "groups": [
                {
                    "id": "producer",
                    "label": "Any service",
                    "source": s_task_queue_task,
                },
                {
                    "id": "transport",
                    "label": "Queue",
                    "source": s_task_queue_spec,
                },
                {
                    "id": "worker",
                    "label": "Worker process (DEPLOY_TARGET)",
                    "source": s_task_listener_run,
                },
                {
                    "id": "run",
                    "label": "Execution",
                    "source": s_task_handler_class,
                },
                {
                    "id": "deps",
                    "label": "Dependencies",
                    "source": s_pg,
                },
            ],
            "nodes": [
                N(
                    "caller",
                    "queue_task()",
                    "producer",
                    s_task_queue_task,
                    [
                        D(
                            "records a task_run row, then sends task JSON to SQS",
                            s_create_queued,
                            s_send,
                        ),
                        D("each TaskType names its queue type", s_task_type_queue),
                    ],
                ),
                N(
                    "task_run",
                    "task_run (Postgres)",
                    "producer",
                    s_task_table,
                    [],
                    kind="store",
                ),
                N(
                    "sqs",
                    "SQS queue per queue type",
                    "transport",
                    s_task_queue_spec,
                    [
                        D(
                            "QueueCatalogEntry: queue URL env var + listener limits",
                            s_task_queue_spec,
                        )
                    ],
                    kind="queue",
                    many=True,
                ),
                N(
                    "listener",
                    "SQS listener (listeners/base.py)",
                    "worker",
                    s_task_listener_run,
                    [
                        D(
                            "DEPLOY_TARGET picks queue types to poll",
                            s_listener_queue_configs,
                        ),
                        D(
                            "drops message if task_run already COMPLETED/STOPPED/SKIPPED",
                            s_listener_drop_states,
                        ),
                    ],
                ),
                N(
                    "handler",
                    "TaskHandler",
                    "run",
                    s_task_handler_class,
                    [
                        D("resolves executor", s_task_executor_resolution),
                        D("updates task_run state when done", s_task_final_state),
                    ],
                ),
                N(
                    "registry",
                    "Executor registry",
                    "run",
                    s_executor_registry,
                    [
                        D(
                            "TASK_EXECUTORS[task_type.task_type](task)",
                            s_executor_dispatch,
                        )
                    ],
                ),
                N(
                    "executor",
                    "TaskExecutor.execute()",
                    "run",
                    s_task_executor_execute,
                    [
                        D(
                            "GoogleWorkspaceAdminCreateTaskExecutor",
                            s_workspace_admin_executor,
                        ),
                        D("EmailIntegrationTaskExecutor", s_email_executor),
                        D(
                            "CalendarExtractionCoordinatorTaskExecutor",
                            s_calendar_executor,
                        ),
                        D(
                            "DataExtractionCoordinatorExecutor",
                            s_data_extraction_executor,
                        ),
                    ],
                    many=True,
                ),
                N(
                    "pg",
                    "Postgres",
                    "deps",
                    s_pg,
                    [],
                    kind="store",
                ),
                N(
                    "ext",
                    "External APIs",
                    "deps",
                    s_ascii_int,
                    [],
                    kind="external",
                ),
            ],
            "edges": [
                E("caller", "task_run", "insert QUEUED", s_create_queued),
                E("caller", "sqs", "send task JSON", s_send),
                E("sqs", "listener", "long-poll", s_poll),
                E("listener", "task_run", "check state", s_state_check),
                E(
                    "listener",
                    "handler",
                    "execute_task",
                    S(
                        "backend/src/tasks/listeners/base.py",
                        "from tasks.task_handler import execute_task",
                    ),
                ),
                E(
                    "handler",
                    "registry",
                    "get_executor",
                    s_task_executor_resolution,
                ),
                E("registry", "executor", "construct", s_executor_dispatch),
                E("executor", "pg", "business logic", s_task_pg_write),
                E("executor", "ext", "provider calls", s_task_provider_call),
                E(
                    "handler",
                    "task_run",
                    "final state",
                    s_task_final_state,
                ),
            ],
        },
    },
    {
        "id": "integration-layer",
        "title": "Integration layer",
        "notes": [
            C(
                "Based on ascii_architecture.md §5 (Integration Layer Architecture), "
                "corrected",
                s_ascii_int,
            ),
            C(
                "Doc lists 4 shims; the current shim packages are shown",
                s_auth0_class,
                s_kv_store,
                s_queue_shim,
            ),
            C(
                "SQS is consumed by the task listeners; the doc showed no consumer",
                s_listener_receive,
            ),
        ],
        "block": {
            "groups": [
                {"id": "business", "label": "Business logic", "source": s_rn},
                {
                    "id": "libs",
                    "label": "rox-core libraries",
                    "source": s_app_calendar,
                },
                {
                    "id": "ext_int",
                    "label": "ext_integrations/",
                    "source": s_google_calendar,
                    "parent": "libs",
                },
                {
                    "id": "shims",
                    "label": "shim_layer/",
                    "source": s_kv_store,
                    "parent": "libs",
                },
                {
                    "id": "outside",
                    "label": "Outside services",
                    "source": s_sqs_client,
                },
                {
                    "id": "providers",
                    "label": "Third-party APIs",
                    "source": s_google_calendar,
                    "parent": "outside",
                },
                {
                    "id": "aws",
                    "label": "AWS + infra",
                    "source": s_dynamo_client,
                    "parent": "outside",
                },
                {
                    "id": "workers",
                    "label": "Task workers",
                    "source": s_listener_receive,
                },
            ],
            "nodes": [
                N(
                    "app",
                    "rox_core/api, chat, tasks",
                    "business",
                    s_rn,
                    [],
                ),
                N(
                    "listener",
                    "SQS listeners",
                    "workers",
                    s_listener_receive,
                    [
                        D("polls its queue", s_listener_receive),
                        D(
                            "hands each task to TaskHandler (Figure 2)",
                            s_task_handler_class,
                        ),
                    ],
                    many=True,
                ),
                N(
                    "calendar_email",
                    "Calendar & email",
                    "ext_int",
                    s_google_calendar,
                    [
                        D("Google Calendar client", s_google_calendar),
                        D("Gmail Google API client", s_gmail_client),
                        D("Outlook service", s_outlook_service),
                    ],
                ),
                N(
                    "messaging",
                    "Messaging",
                    "ext_int",
                    s_slack_sdk,
                    [
                        D("Slack WebClient SDK", s_slack_sdk),
                        D("Microsoft Teams MSAL client", s_microsoft_sdk),
                        D("Unipile integration", s_unipile),
                    ],
                ),
                N(
                    "crm",
                    "CRM",
                    "ext_int",
                    s_crm_parser,
                    [
                        D("HubSpot field parser", s_crm_parser),
                        D("HubSpot SDK", s_hubspot_sdk),
                        D("Salesforce SDK", s_salesforce_sdk),
                    ],
                ),
                N(
                    "telephony",
                    "Telephony & recorders",
                    "ext_int",
                    s_twilio_sdk,
                    [
                        D("Twilio REST client", s_twilio_sdk),
                        D("Orum service", s_orum_service),
                        D("Call recorder event service", s_call_recorder),
                    ],
                ),
                N(
                    "data_providers",
                    "Data providers",
                    "ext_int",
                    s_apollo_api,
                    [
                        D("Apollo API base URL", s_apollo_api),
                        D("Cognism client", s_cognism),
                    ],
                ),
                N(
                    "auth_secrets",
                    "Auth & secrets",
                    "shims",
                    s_auth0_class,
                    [
                        D("Auth0 shim", s_auth0_class),
                        D("AWS KMS shim", s_kms_class),
                        D("Secrets Manager client", s_secrets_factory),
                    ],
                ),
                N(
                    "store_shims",
                    "Stores",
                    "shims",
                    s_dynamo_client,
                    [
                        D("DynamoDB client", s_dynamo_client),
                        D("S3 client", s_s3_client),
                        D("Redis client SDK", s_redis_sdk),
                        D("KV store interface", s_kv_store),
                    ],
                ),
                N(
                    "queue_shims",
                    "Queues",
                    "shims",
                    s_queue_shim,
                    [
                        D("Queue abstraction", s_queue_shim),
                        D("SQS queue shim", s_sqs_shim),
                    ],
                ),
                N(
                    "billing",
                    "Billing",
                    "shims",
                    s_stripe_sdk,
                    [
                        D("Stripe SDK", s_stripe_sdk),
                        D("Rillet API endpoint", s_rillet_api),
                    ],
                ),
                N(
                    "obs",
                    "Observability & push",
                    "shims",
                    s_observability,
                    [
                        D("Unified observability layer", s_observability),
                        D("SNS push shim", s_sns_push),
                    ],
                ),
                N(
                    "google_microsoft",
                    "Google / Microsoft",
                    "providers",
                    s_google_calendar,
                    [
                        D("Google Calendar API", s_google_calendar),
                        D("Gmail API client", s_gmail_client),
                        D("Outlook service", s_outlook_service),
                    ],
                    kind="external",
                ),
                N(
                    "slack_api",
                    "Slack",
                    "providers",
                    s_slack_sdk,
                    [D("Slack WebClient SDK", s_slack_sdk)],
                    kind="external",
                ),
                N(
                    "crm_apis",
                    "Salesforce / HubSpot",
                    "providers",
                    s_salesforce_sdk,
                    [
                        D("Salesforce SDK", s_salesforce_sdk),
                        D("HubSpot SDK", s_hubspot_sdk),
                    ],
                    kind="external",
                ),
                N(
                    "twilio_api",
                    "Twilio",
                    "providers",
                    s_twilio_sdk,
                    [D("Twilio REST client", s_twilio_sdk)],
                    kind="external",
                ),
                N(
                    "stripe_api",
                    "Stripe",
                    "providers",
                    s_stripe_sdk,
                    [D("Stripe SDK", s_stripe_sdk)],
                    kind="external",
                ),
                N(
                    "auth0_api",
                    "Auth0",
                    "providers",
                    s_auth0_management,
                    [D("Auth0 Management API", s_auth0_management)],
                    kind="external",
                ),
                N(
                    "data_vendors",
                    "Data vendors",
                    "providers",
                    s_apollo_api,
                    [
                        D("Apollo API", s_apollo_api),
                        D("Cognism client", s_cognism),
                    ],
                    kind="external",
                ),
                N(
                    "dynamodb",
                    "DynamoDB",
                    "aws",
                    s_dynamo_client,
                    [D("boto3 DynamoDB client", s_dynamo_client)],
                    kind="store",
                ),
                N(
                    "s3",
                    "S3",
                    "aws",
                    s_s3_client,
                    [D("boto3 S3 client", s_s3_client)],
                    kind="store",
                ),
                N(
                    "sqs_aws",
                    "SQS",
                    "aws",
                    s_sqs_client,
                    [
                        D("boto3 SQS client", s_sqs_client),
                        D("one queue per queue type", s_listener_queue_configs),
                    ],
                    kind="queue",
                    many=True,
                ),
                N(
                    "redis",
                    "Redis",
                    "aws",
                    s_redis_sdk,
                    [D("redis-py client", s_redis_sdk)],
                    kind="store",
                ),
                N(
                    "kms_secrets",
                    "KMS / Secrets Manager",
                    "aws",
                    s_kms_client,
                    [
                        D("boto3 KMS client", s_kms_client),
                        D("Secrets Manager client", s_secrets_client),
                    ],
                    kind="store",
                ),
            ],
            "edges": [
                E("app", "ext_int", "calls", s_app_calendar),
                E("app", "shims", "calls", s_app_stores),
                E(
                    "calendar_email",
                    "google_microsoft",
                    "Google Calendar API",
                    s_google_calendar,
                ),
                E(
                    "messaging",
                    "google_microsoft",
                    "Microsoft Teams API",
                    s_microsoft_sdk,
                ),
                E("messaging", "slack_api", "Slack SDK", s_slack_sdk),
                E("crm", "crm_apis", "Salesforce API", s_salesforce_sdk),
                E("telephony", "twilio_api", "Twilio client", s_twilio_sdk),
                E("data_providers", "data_vendors", "Apollo API", s_apollo_api),
                E(
                    "auth_secrets",
                    "auth0_api",
                    "Auth0 Management API",
                    s_auth0_management,
                ),
                E("auth_secrets", "kms_secrets", "AWS KMS client", s_kms_client),
                E("store_shims", "dynamodb", "DynamoDB client", s_dynamo_client),
                E("store_shims", "s3", "S3 client", s_s3_client),
                E("store_shims", "redis", "Redis client", s_redis_sdk),
                E("queue_shims", "sqs_aws", "SQS client", s_sqs_client),
                E("billing", "stripe_api", "Stripe client", s_stripe_sdk),
                E("sqs_aws", "listener", "polled by", s_listener_receive),
            ],
        },
    },
    {
        "id": "connect-google-workspace",
        "title": "Connecting an integration (Google Workspace admin)",
        "notes": [
            C(
                "Based on ascii_architecture.md §7 (Calendar Integration), corrected",
                S(ASCII, "## 7. Data Flow: Calendar Integration Example"),
            ),
            C(
                "Runs on the INTEGRATION queue/worker, not AGENT/EXTRACTIONAGENT",
                s_google_workspace_queue_type,
            ),
            C(
                "/integrations/gcal is only the OAuth callback",
                s_gcal_callback,
            ),
            C(
                "The periodic CalendarExtractionCoordinator is a separate scheduled "
                "refresh",
                s_calendar_refresh_schedule,
            ),
        ],
        "block": {
            "groups": [
                {"id": "client", "label": "Client", "source": s_web},
                {
                    "id": "http",
                    "label": "INTERACTION",
                    "source": s_integrations_route,
                },
                {"id": "q", "label": "Queue", "source": s_google_workspace_queue_type},
                {
                    "id": "iw",
                    "label": "INTEGRATION worker",
                    "source": s_workspace_admin_executor,
                },
                {
                    "id": "fanout_col",
                    "label": "Fan-out",
                    "source": s_integration_fanout,
                },
                {
                    "id": "runners",
                    "label": "Runners",
                    "source": s_temporal_calendar_workflow,
                },
                {
                    "id": "google_col",
                    "label": "Google APIs",
                    "source": s_google_calendar,
                },
            ],
            "nodes": [
                N("web", "web app", "client", s_web, []),
                N(
                    "post",
                    "POST /integrations",
                    "http",
                    s_integrations_route,
                    [
                        D(
                            "create_integration(request)",
                            s_create_integration_call,
                        )
                    ],
                ),
                N(
                    "create",
                    "_create_integration()",
                    "http",
                    s_create_integration,
                    [
                        D(
                            "caches org workspace-admin flag",
                            s_workspace_admin_cache,
                        ),
                        D(
                            "queue_task(GoogleWorkspaceAdminCreateTask)",
                            s_workspace_admin_task,
                            s_workspace_admin_queue_call,
                        ),
                    ],
                ),
                N(
                    "integration_tbl",
                    "integration (Postgres)",
                    "http",
                    s_integration_table,
                    [],
                    kind="store",
                ),
                N(
                    "iq",
                    "INTEGRATION queue",
                    "q",
                    s_google_workspace_queue_type,
                    [],
                    kind="queue",
                ),
                N(
                    "gwa",
                    "GoogleWorkspaceAdminCreateTaskExecutor",
                    "iw",
                    s_workspace_admin_executor,
                    [
                        D(
                            "loads integration, checks type",
                            s_workspace_admin_type_check,
                        ),
                        D(
                            "creates per-user CAL + EMAIL integrations",
                            s_google_workspace_user_integrations,
                        ),
                    ],
                ),
                N(
                    "fanout",
                    "fan_out_tasks_for_integrations()",
                    "fanout_col",
                    s_integration_fanout,
                    [
                        D(
                            "initialize calendar integrations",
                            s_initialize_calendar_integrations,
                        ),
                        D(
                            "org-wide calendar: Temporal workflow, else SQS tasks",
                            s_temporal_calendar_start,
                            s_sqs_calendar_tasks,
                        ),
                        D(
                            "org-wide email: Temporal workflow, else SQS tasks",
                            s_temporal_email_start,
                            s_sqs_calendar_tasks,
                        ),
                    ],
                    many=True,
                ),
                N(
                    "temporal",
                    "Temporal (org-wide calendar/email workflows)",
                    "runners",
                    s_temporal_calendar_workflow,
                    [],
                    kind="queue",
                ),
                N(
                    "sqs_tasks",
                    "SQS extraction tasks",
                    "runners",
                    s_sqs_calendar_tasks,
                    [],
                    kind="queue",
                ),
                N(
                    "google",
                    "Google Calendar / Gmail APIs",
                    "google_col",
                    s_google_calendar,
                    [],
                    kind="external",
                ),
            ],
            "edges": [
                E("web", "post", "connect", s_integrations_route),
                E(
                    "post",
                    "create",
                    "create_integration",
                    s_create_integration_call,
                ),
                E(
                    "create",
                    "integration_tbl",
                    "save row",
                    s_integration_row_save,
                ),
                E(
                    "create",
                    "iq",
                    "queue_task",
                    s_workspace_admin_queue_call,
                ),
                E(
                    "iq",
                    "gwa",
                    "INTEGRATION worker",
                    s_google_workspace_queue_type,
                ),
                E(
                    "gwa",
                    "integration_tbl",
                    "per-user rows",
                    s_google_workspace_user_integrations,
                ),
                E(
                    "gwa",
                    "fanout",
                    "fan_out_tasks_for_new_integrations",
                    s_new_integration_fanout_call,
                ),
                E(
                    "fanout",
                    "temporal",
                    "start workflow",
                    s_temporal_calendar_start,
                ),
                E(
                    "fanout",
                    "sqs_tasks",
                    "fallback / direct",
                    s_sqs_calendar_tasks,
                ),
                E(
                    "temporal",
                    "google",
                    "fetch events / mail",
                    s_google_calendar,
                ),
                E(
                    "sqs_tasks",
                    "google",
                    "fetch events / mail",
                    s_google_calendar,
                ),
            ],
        },
    },
    {
        "id": "chat-turn",
        "title": "A chat turn (resumable stream)",
        "notes": [
            C(
                "Condensed from chat/background_execution/README.md; full flow on the "
                "chat page",
                s_chat_mermaid,
            ),
            C(
                "README checked against stream_runner.py at this commit",
                s_stream_hook_type,
            ),
        ],
        "block": {
            "groups": [
                {"id": "client", "label": "Browser", "source": s_web},
                {
                    "id": "chat",
                    "label": "CHAT service (FastAPI)",
                    "source": s_chat_post_route,
                },
                {
                    "id": "bg",
                    "label": "Background asyncio task",
                    "source": s_chat_runner_run,
                },
                {"id": "stores", "label": "Stores", "source": s_conv_table},
            ],
            "nodes": [
                N("web", "web app", "client", s_web, []),
                N(
                    "post_msg",
                    "POST /message/{conversation_id}",
                    "chat",
                    s_chat_post_route,
                    [],
                ),
                N(
                    "start_turn",
                    "start_turn()",
                    "chat",
                    s_chat_start_turn,
                    [
                        D(
                            "initialize Redis stream (new UUID)",
                            s_chat_stream_initialize,
                        ),
                        D(
                            "reserve stream_id on conversation (fail_if_active)",
                            s_chat_fail_if_active,
                        ),
                        D(
                            "pre_spawn_hook: persist user message",
                            s_chat_persist_hook,
                        ),
                        D("spawn producer task", s_chat_spawn_task),
                    ],
                ),
                N(
                    "producer",
                    "ConversationStreamRunner.run()",
                    "bg",
                    s_chat_runner_run,
                    [
                        D("heartbeats to Redis", s_chat_heartbeat),
                        D("append each event", s_append),
                    ],
                ),
                N(
                    "agent",
                    "agent_event_generator → RoxRunner turn loop",
                    "bg",
                    s_agent_event_generator,
                    [
                        D(
                            "LLM call + tools per turn",
                            s_rox_runner_streaming,
                        )
                    ],
                ),
                N(
                    "get_stream",
                    "GET /message/stream/{conversation_id}",
                    "chat",
                    s_chat_get_route,
                    [],
                ),
                N(
                    "conv",
                    "conversation (Postgres)",
                    "stores",
                    s_conv_table,
                    [],
                    kind="store",
                ),
                N(
                    "redis",
                    "Redis stream",
                    "stores",
                    s_redis,
                    [],
                    kind="store",
                ),
            ],
            "edges": [
                E("web", "post_msg", "send", s_chat_post_route),
                E("post_msg", "start_turn", "start_turn", s_chat_start_turn),
                E(
                    "start_turn",
                    "redis",
                    "init stream",
                    s_chat_stream_initialize,
                ),
                E(
                    "start_turn",
                    "conv",
                    "reserve stream_id",
                    s_chat_fail_if_active,
                ),
                E(
                    "start_turn",
                    "producer",
                    "asyncio.create_task",
                    s_chat_spawn_task,
                ),
                E(
                    "producer",
                    "agent",
                    "iterate events",
                    s_agent_event_generator,
                ),
                E(
                    "producer",
                    "redis",
                    "append / heartbeat",
                    s_chat_append_heartbeat,
                ),
                E("web", "get_stream", "SSE", s_sse),
                E(
                    "get_stream",
                    "redis",
                    "read (resumable)",
                    s_chat_redis_read,
                ),
            ],
        },
    },
]

page = {
    "id": "rox-core",
    "title": "rox-core",
    "kind": "root",
    "commit": COMMIT,
    "parent": None,
    "paths": ["."],
    "tldr": tldr,
    "block": {"groups": groups, "nodes": nodes, "edges": edges},
    "block_figures": block_figures,
    "data": data,
    "sequences": sequences,
    "states": [],
    "related": related,
}

root_nodes_by_id = {node["id"]: node for node in nodes}
component_catalog = [
    {
        "id": "web",
        "root_id": "web",
        "kind": "client",
        "column": "Callers",
        "match": {"web_files": True},
    },
    {
        "id": "http_clients",
        "root_id": "web",
        "label": "HTTP clients",
        "kind": "client",
        "column": "Callers",
        "match": {},
    },
    {
        "id": "provider_push",
        "label": "Provider push",
        "kind": "external",
        "shape": "external",
        "many": False,
        "column": "Callers",
        "match": {"implied_by": "WEBHOOK endpoint"},
        "source": s_init_webhook,
    },
    {
        "id": "interaction",
        "root_id": "interaction",
        "kind": "service",
        "column": "HTTP services",
        "match": {
            "deploy_targets": ["INTERACTION"],
            "queue_type_classes": ["InteractionQueueType"],
        },
    },
    {
        "id": "webhook",
        "root_id": "interaction",
        "label": "WEBHOOK (Flask)",
        "kind": "service",
        "column": "HTTP services",
        "match": {"deploy_targets": ["WEBHOOK"]},
        "extra_sources": [s_init_webhook],
    },
    {
        "id": "chat",
        "root_id": "chat",
        "kind": "service",
        "column": "HTTP services",
        "match": {"path_prefixes": ["backend/src/chat/"]},
    },
    {
        "id": "public_api",
        "root_id": "public_api",
        "kind": "service",
        "column": "HTTP services",
        "match": {"path_prefixes": ["backend/src/public_api/"]},
    },
    {
        "id": "mcp",
        "root_id": "mcp",
        "kind": "service",
        "column": "HTTP services",
        "match": {"path_prefixes": ["backend/src/external_mcp/"]},
    },
    {
        "id": "temporal",
        "root_id": "temporal_workers",
        "label": "Temporal workflows",
        "kind": "queue",
        "column": "Workflows",
        "match": {
            "worker_kinds": ["temporal_workflow", "temporal_activity"],
            "workflow_start_callers": True,
        },
    },
    {
        "id": "sqs",
        "root_id": "sqs",
        "label": "SQS queues",
        "kind": "queue",
        "column": "Queues",
        "match": {"task_producers_or_consumers": True},
    },
    {
        "id": "agent_workers",
        "root_id": "agent_workers",
        "kind": "service(many)",
        "column": "Background workers",
        "match": {
            "deploy_targets": [
                "AGENT",
                "REALTIMEAGENT",
                "EXTRACTIONAGENT",
                "ASYNC_AGENT",
            ]
        },
    },
    {
        "id": "data_workers",
        "root_id": "data_workers",
        "kind": "service(many)",
        "column": "Background workers",
        "match": {"deploy_targets": ["SOR", "BATCH", "BACKFILL", "INTEGRATION"]},
    },
    {
        "id": "outreach_sched",
        "root_id": "outreach_sched",
        "kind": "service(many)",
        "column": "Background workers",
        "match": {"deploy_targets": ["OUTREACH", "WORKFLOWSCHEDULER"]},
    },
    {
        "id": "postgres",
        "root_id": "postgres",
        "label": "PostgreSQL",
        "kind": "store",
        "column": "Stores & provider APIs",
        "match": {"table_access": True},
    },
    {
        "id": "llm",
        "root_id": "llm",
        "kind": "external",
        "column": "Stores & provider APIs",
        "match": {"external_services": ["OpenAI", "Anthropic", "LiteLLM"]},
    },
    {
        "id": "crm",
        "root_id": "crm",
        "kind": "external",
        "column": "Stores & provider APIs",
        "match": {
            "external_services": [
                "Salesforce",
                "HubSpot",
                "Google APIs",
                "Slack",
                "Microsoft identity",
            ]
        },
    },
    {
        "id": "twilio",
        "root_id": "twilio",
        "kind": "external",
        "column": "Stores & provider APIs",
        "match": {"external_services": ["Twilio"]},
    },
]
for catalog_entry in component_catalog:
    root_id = catalog_entry.pop("root_id", None)
    root_node = root_nodes_by_id.get(root_id)
    if root_node is not None:
        catalog_entry.setdefault("label", root_node["label"])
        catalog_entry["shape"] = root_node["kind"]
        catalog_entry["many"] = root_node["many"]
        catalog_entry["source"] = root_node["source"]
    extra_sources = catalog_entry.pop("extra_sources", [])
    if extra_sources:
        catalog_entry["sources"] = [catalog_entry["source"], *extra_sources]

out = Path(__file__).resolve().parents[1] / "rox-core.json"
out.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n")
print("wrote", out, len(nodes), "nodes", len(edges), "edges")
component_catalog_out = out.parent / "components.json"
component_catalog_out.write_text(
    json.dumps(component_catalog, indent=2, ensure_ascii=False) + "\n"
)
print("wrote", component_catalog_out, len(component_catalog), "components")
