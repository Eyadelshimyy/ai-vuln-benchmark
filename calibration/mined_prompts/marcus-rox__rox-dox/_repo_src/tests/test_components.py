from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rox_dox import cache, components, features
from rox_dox.components import collect_task_facts, extract_file_facts


def _commit_sources(
    tmp_path: Path,
    sources: dict[str, str],
) -> tuple[Path, str]:
    repo = tmp_path / "component-source"
    repo.mkdir()
    for relative_path, source in sources.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", *sources],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add component fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def _component_sources() -> dict[str, str]:
    return {
        "backend/src/rox_core/api/integrations/endpoints.py": (
            "from flask_restx import Namespace as FlaskRestxNamespace\n"
            "\n"
            'integrations_ns = FlaskRestxNamespace(name="integrations", validate=True)\n'
            "\n"
            '@integrations_ns.route("/health")\n'
            "class Health:\n"
            "    def get(self):\n"
            '        return "ok"\n'
            "\n"
            "    def post(self):\n"
            '        return "created"\n'
        ),
        "backend/src/rox_core/api/events/endpoints.py": (
            "from flask_restx import Namespace\n"
            "\n"
            'events_ns = Namespace("events")\n'
            "\n"
            '@events_ns.route("/sync")\n'
            "class Sync:\n"
            "    def put(self):\n"
            '        return "synced"\n'
        ),
        "backend/src/rox_core/api/register.py": (
            'api.add_namespace(integrations_ns, path="/integrations")\n'
            'api.add_namespace(events_ns, path="/events")\n'
            'api.add_namespace(events_ns, path="/v2/events")\n'
        ),
        "backend/src/rox_core/api/fastapi_router.py": (
            "from fastapi import APIRouter\n"
            "\n"
            'router = APIRouter(prefix="/v2/", tags=["integrations"])\n'
            "\n"
            '@router.get("/status")\n'
            "def status():\n"
            '    return {"status": "ok"}\n'
        ),
        "backend/src/rox_core/api/app_routes.py": (
            "from fastapi import FastAPI\n"
            "\n"
            "app = FastAPI()\n"
            "\n"
            '@app.post("/ready")\n'
            "def ready():\n"
            '    return {"ready": True}\n'
        ),
        "backend/src/rox_core/workers/integration_workers.py": (
            "class TaskExecutor:\n"
            "    pass\n"
            "\n"
            "class BaseFooTaskExecutor(TaskExecutor):\n"
            "    pass\n"
            "\n"
            "class Foo(BaseFooTaskExecutor):\n"
            "    pass\n"
            "\n"
            "@workflow.defn\n"
            "class IntegrationWorkflow:\n"
            "    pass\n"
            "\n"
            "@activity.defn\n"
            "def sync_activity():\n"
            "    return None\n"
        ),
        "backend/src/rox_core/services/integrations.py": (
            "import google.oauth2.credentials\nimport requests\nimport httpx\n"
        ),
    }


def test_flask_namespace_prefix_and_nonunique_fallback_are_resolved(
    tmp_path: Path,
) -> None:
    repo, commit = _commit_sources(tmp_path, _component_sources())
    facts = extract_file_facts(
        repo,
        commit,
        [
            "backend/src/rox_core/api/integrations/endpoints.py",
            "backend/src/rox_core/api/events/endpoints.py",
        ],
    )

    integrations = facts["backend/src/rox_core/api/integrations/endpoints.py"]
    assert integrations.api_group == "integrations"
    assert [
        (endpoint.method, endpoint.path, endpoint.handler, endpoint.line)
        for endpoint in integrations.endpoints
    ] == [
        ("GET", "/integrations/health", "Health", 5),
        ("POST", "/integrations/health", "Health", 5),
    ]
    events = facts["backend/src/rox_core/api/events/endpoints.py"]
    assert events.api_group == "events"
    assert [(endpoint.method, endpoint.path) for endpoint in events.endpoints] == [
        ("PUT", "/events/sync")
    ]


def test_fastapi_router_prefix_and_app_decorators_are_extracted(
    tmp_path: Path,
) -> None:
    repo, commit = _commit_sources(tmp_path, _component_sources())
    facts = extract_file_facts(
        repo,
        commit,
        [
            "backend/src/rox_core/api/fastapi_router.py",
            "backend/src/rox_core/api/app_routes.py",
        ],
    )

    router = facts["backend/src/rox_core/api/fastapi_router.py"]
    assert router.api_group == "v2"
    assert [(endpoint.method, endpoint.path) for endpoint in router.endpoints] == [
        ("GET", "/v2/status")
    ]
    app = facts["backend/src/rox_core/api/app_routes.py"]
    assert app.api_group == "app_routes"
    assert [(endpoint.method, endpoint.path) for endpoint in app.endpoints] == [
        ("POST", "/ready")
    ]


def test_task_executor_and_temporal_workers_are_extracted(tmp_path: Path) -> None:
    repo, commit = _commit_sources(tmp_path, _component_sources())
    facts = extract_file_facts(
        repo,
        commit,
        ["backend/src/rox_core/workers/integration_workers.py"],
    )["backend/src/rox_core/workers/integration_workers.py"]

    assert [(worker.name, worker.kind) for worker in facts.workers] == [
        ("Foo", "task_executor"),
        ("sync_activity", "temporal_activity"),
        ("IntegrationWorkflow", "temporal_workflow"),
    ]


def test_dotted_google_module_matches_while_requests_and_httpx_are_ignored(
    tmp_path: Path,
) -> None:
    repo, commit = _commit_sources(tmp_path, _component_sources())
    facts = extract_file_facts(
        repo,
        commit,
        ["backend/src/rox_core/services/integrations.py"],
    )["backend/src/rox_core/services/integrations.py"]

    assert [(call.service, call.module, call.line) for call in facts.externals] == [
        ("Google APIs", "google.oauth2.credentials", 1)
    ]


def test_extraction_reads_the_requested_commit_not_working_tree(tmp_path: Path) -> None:
    repo, commit = _commit_sources(tmp_path, _component_sources())
    path = "backend/src/rox_core/api/integrations/endpoints.py"
    (repo / path).write_text("not a namespace\n", encoding="utf-8")

    facts = extract_file_facts(repo, commit, [path])[path]

    assert facts.api_group == "integrations"
    assert len(facts.endpoints) == 2


def test_extraction_is_deterministic_across_hash_seeds(tmp_path: Path) -> None:
    repo, commit = _commit_sources(tmp_path, _component_sources())
    paths = sorted(_component_sources())
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from rox_dox.components import extract_file_facts\n"
        "facts = extract_file_facts(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])\n"
        "print(json.dumps({key: value.model_dump(mode='json') for key, value in facts.items()}, "
        "sort_keys=True, separators=(',', ':')))\n"
    )
    outputs = []
    for seed in ("1", "137"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1] / "src"),
                env.get("PYTHONPATH", ""),
            ]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(repo),
                commit,
                *paths,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])


def test_task_types_consumers_and_producers_are_extracted(tmp_path: Path) -> None:
    sources = {
        "backend/src/tasks/types.py": (
            "from enum import Enum\n"
            "\n"
            "class TaskTypeInfo:\n"
            "    pass\n"
            "\n"
            "class TaskType(Enum):\n"
            "    NOTIFICATION_SENDER = TaskTypeInfo(\n"
            "        queue_type=RealtimeAgentQueueType.NOTIFICATION_SENDER\n"
            "    )\n"
        ),
        "backend/src/tasks/executor_registry.py": (
            "from .types import TaskType\n"
            "\n"
            "TASK_EXECUTORS = {\n"
            "    TaskType.NOTIFICATION_SENDER: NotificationSenderTaskExecutor,\n"
            "}\n"
        ),
        "backend/src/tasks/notification_sender.py": (
            "class TaskExecutor:\n"
            "    pass\n"
            "\n"
            "class NotificationSenderTaskExecutor(TaskExecutor):\n"
            "    pass\n"
        ),
        "backend/src/tasks/producer.py": (
            "def enqueue(payload):\n"
            "    submit(payload, task_type=TaskType.NOTIFICATION_SENDER.value.name)\n"
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)
    facts = extract_file_facts(repo, commit, sorted(sources))

    assert [
        (task.name, task.queue_type, task.path, task.line)
        for task in facts["backend/src/tasks/types.py"].task_types
    ] == [
        (
            "NOTIFICATION_SENDER",
            "RealtimeAgentQueueType.NOTIFICATION_SENDER",
            "backend/src/tasks/types.py",
            7,
        )
    ]
    assert [
        (consumer.task_type, consumer.executor, consumer.path, consumer.line)
        for consumer in facts["backend/src/tasks/executor_registry.py"].task_consumers
    ] == [
        (
            "NOTIFICATION_SENDER",
            "NotificationSenderTaskExecutor",
            "backend/src/tasks/executor_registry.py",
            4,
        )
    ]
    assert [
        (producer.task_type, producer.path, producer.line)
        for producer in facts["backend/src/tasks/producer.py"].task_producers
    ] == [
        (
            "NOTIFICATION_SENDER",
            "backend/src/tasks/producer.py",
            2,
        )
    ]
    assert [
        (worker.name, worker.line)
        for worker in facts["backend/src/tasks/notification_sender.py"].workers
    ] == [("NotificationSenderTaskExecutor", 4)]


def test_namespace_registration_records_webhook_deploy_target(
    tmp_path: Path,
) -> None:
    sources = {
        "backend/src/rox_core/api/register_namespaces.py": (
            'if deploy_target == "WEBHOOK":\n'
            '    api.add_namespace(webhook_ns, path="/webhooks")\n'
            'elif deploy_target == "INTERACTION":\n'
            '    api.add_namespace(other_ns, path="/api")\n'
        ),
        "backend/src/rox_core/api/webhook/endpoints.py": (
            "from flask_restx import Namespace\n"
            'webhook_ns = Namespace("webhook")\n'
            '@webhook_ns.route("/events")\n'
            "class Events:\n"
            "    def post(self):\n"
            '        return "ok"\n'
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)
    endpoints = extract_file_facts(
        repo,
        commit,
        ["backend/src/rox_core/api/webhook/endpoints.py"],
    )["backend/src/rox_core/api/webhook/endpoints.py"].endpoints

    assert [(endpoint.path, endpoint.deploy_target) for endpoint in endpoints] == [
        ("/webhooks/events", "WEBHOOK")
    ]


def test_webhook_blueprint_routes_are_extracted_with_methods(
    tmp_path: Path,
) -> None:
    sources = {
        "backend/src/rox_core/__init__.py": (
            "def create_app():\n"
            '    if deploy_target_for_blueprints == "WEBHOOK":\n'
            "        from rox_core.api.webhook.routes import webhook_bp\n"
            "        app.register_blueprint(webhook_bp)\n"
            "    else:\n"
            "        pass\n"
        ),
        "backend/src/rox_core/api/webhook/routes.py": (
            "from flask import Blueprint\n"
            'webhook_bp = Blueprint("provider", __name__, url_prefix="/provider")\n'
            '@webhook_bp.route("/events", methods=["POST", "GET"])\n'
            "def receive_event():\n"
            "    return None\n"
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)
    endpoints = extract_file_facts(
        repo,
        commit,
        ["backend/src/rox_core/api/webhook/routes.py"],
    )["backend/src/rox_core/api/webhook/routes.py"].endpoints

    assert [
        (endpoint.method, endpoint.path, endpoint.deploy_target)
        for endpoint in endpoints
    ] == [
        ("GET", "/provider/events", "WEBHOOK"),
        ("POST", "/provider/events", "WEBHOOK"),
    ]


def test_webhook_blueprint_registration_maps_namespace_deploy_target(
    tmp_path: Path,
) -> None:
    endpoint_path = "backend/src/rox_core/api/msteams_agent/endpoints.py"
    sources = {
        "backend/src/rox_core/__init__.py": (
            'if deploy_target_for_blueprints == "WEBHOOK":\n'
            "    from rox_core.api.msteams_agent.webhook_bp import "
            "msteams_webhook_bp\n"
            "    app.register_blueprint(msteams_webhook_bp)\n"
        ),
        "backend/src/rox_core/api/msteams_agent/webhook_bp.py": (
            "from flask import Blueprint\n"
            "from flask_restx import Api\n"
            "from .endpoints import msteams_agent_ns\n"
            '\nwebhook_bp = Blueprint("msteams", __name__, url_prefix="/webhooks")\n'
            "api = Api(webhook_bp)\n"
            'api.add_namespace(msteams_agent_ns, path="/msteams_agent")\n'
        ),
        endpoint_path: (
            "from flask_restx import Namespace\n"
            'msteams_agent_ns = Namespace("msteams_agent")\n'
            '\n@msteams_agent_ns.route("/events")\n'
            "class Events:\n"
            "    def post(self):\n"
            '        return "ok"\n'
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)

    endpoints = extract_file_facts(repo, commit, [endpoint_path])[
        endpoint_path
    ].endpoints

    assert [(endpoint.path, endpoint.deploy_target) for endpoint in endpoints] == [
        ("/msteams_agent/events", "WEBHOOK")
    ]


def test_enum_member_queue_class_maps_to_configured_deploy_target(
    tmp_path: Path,
) -> None:
    sources = {
        "backend/src/util/listener_utils.py": (
            "QUEUE_CONFIGS = {\n"
            '    "INTERACTION": ("interaction task", "InteractionQueueType"),\n'
            "}\n"
        ),
        "backend/src/tasks/types.py": (
            "from enum import Enum\n"
            "class TaskTypeInfo:\n"
            "    pass\n"
            "class TaskType(Enum):\n"
            "    MAPPED = TaskTypeInfo(\n"
            "        queue_type=InteractionQueueType.MEMBER,\n"
            "        supports_dynamic_routing=True,\n"
            "        dynamic_routing_enum=UnknownQueueType,\n"
            "    )\n"
            "    UNKNOWN = TaskTypeInfo(queue_type=UnknownQueueType.MEMBER)\n"
            "    MALFORMED = TaskTypeInfo(queue_type=make_queue())\n"
            "    MISSING = TaskTypeInfo()\n"
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)
    facts = extract_file_facts(repo, commit, ["backend/src/tasks/types.py"])
    task_facts = facts["backend/src/tasks/types.py"].task_types
    tasks_by_name = {task.name: task for task in task_facts}

    assert (
        tasks_by_name["MAPPED"].queue_type,
        tasks_by_name["MAPPED"].queue_class,
        tasks_by_name["MAPPED"].deploy_target,
    ) == (
        "InteractionQueueType.MEMBER",
        "InteractionQueueType",
        "INTERACTION",
    )
    assert tasks_by_name["UNKNOWN"].deploy_target is None
    assert tasks_by_name["MALFORMED"].queue_class is None
    assert "MISSING" not in tasks_by_name
    summary = collect_task_facts(facts)
    assert summary.unmapped_queue_type_count == 2
    assert summary.unmapped_queue_type_values == [
        "UnknownQueueType.MEMBER",
        "make_queue()",
    ]


def test_temporal_workflow_start_calls_are_recorded(
    tmp_path: Path,
) -> None:
    sources = {
        "backend/src/rox_core/workers/start_workflow.py": (
            "async def start(client):\n"
            "    await client.start_workflow(MyWorkflow.run, id='one')\n"
            "    await client.execute_workflow(MyWorkflow.run, id='two')\n"
        )
    }
    repo, commit = _commit_sources(tmp_path, sources)
    facts = extract_file_facts(
        repo,
        commit,
        ["backend/src/rox_core/workers/start_workflow.py"],
    )["backend/src/rox_core/workers/start_workflow.py"]

    assert facts.workflow_starts == [2, 3]


def test_parallel_and_serial_file_facts_are_equal(tmp_path: Path, monkeypatch) -> None:
    sources = {
        f"backend/src/workers/module_{index:02d}.py": (
            "from temporalio import workflow\n"
            "\n"
            "@workflow.defn\n"
            f"class Workflow{index}:\n"
            "    async def run(self):\n"
            "        await client.start_workflow(Workflow)\n"
        )
        for index in range(60)
    }
    repo, commit = _commit_sources(tmp_path, sources)
    paths = sorted(sources)

    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "parallel-cache")
    parallel = extract_file_facts(repo, commit, paths)
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "serial-cache")
    monkeypatch.setattr(components, "PARALLEL_MIN_FILES", len(paths))
    serial = extract_file_facts(repo, commit, paths)

    assert len(paths) > features.PARALLEL_MIN_FILES
    assert parallel == serial
    assert all(facts.workers for facts in parallel.values())
