from rox_dox.components import (
    Endpoint,
    ExternalCall,
    FileFacts,
    TaskConsumer,
    TaskProducer,
    TaskType,
    Worker,
)
from rox_dox.features import FeatureEvidence, FeatureFile
from rox_dox.model import CodeSource, Edge, Node
from rox_dox.projection import (
    _deduplicate_edges,
    _flow_notes,
    _prune_early_edges,
    project_component_diagram,
)


def _entry(
    component_id: str,
    label: str,
    kind: str,
    column: str,
    shape: str,
    match: dict,
) -> dict:
    return {
        "id": component_id,
        "label": label,
        "kind": kind,
        "column": column,
        "shape": shape,
        "match": match,
        "source": {"path": f"catalog/{component_id}.py", "lines": [1, 1]},
    }


def _runtime_catalog() -> list[dict]:
    return [
        _entry("web", "web app", "client", "Callers", "component", {"web_files": True}),
        _entry("http_clients", "HTTP clients", "client", "Callers", "component", {}),
        _entry(
            "provider_push",
            "Provider push",
            "external",
            "Callers",
            "external",
            {"implied_by": "WEBHOOK endpoint"},
        ),
        _entry(
            "interaction",
            "INTERACTION (Flask)",
            "service",
            "HTTP services",
            "component",
            {"deploy_targets": ["INTERACTION"]},
        ),
        _entry(
            "webhook",
            "WEBHOOK (Flask)",
            "service",
            "HTTP services",
            "component",
            {"deploy_targets": ["WEBHOOK"]},
        ),
        _entry(
            "temporal",
            "Temporal workflows",
            "queue",
            "Workflows",
            "component",
            {"worker_kinds": ["temporal_workflow", "temporal_activity"]},
        ),
        _entry("sqs", "SQS queues", "queue", "Queues", "queue", {}),
        _entry(
            "data_workers",
            "INTEGRATION workers",
            "service(many)",
            "Background workers",
            "component",
            {
                "deploy_targets": ["INTEGRATION"],
                "queue_type_classes": ["IntegrationQueueType"],
            },
        ),
        _entry("postgres", "PostgreSQL", "store", "Stores", "store", {}),
        _entry(
            "crm",
            "CRM + workspace APIs",
            "external",
            "Provider APIs",
            "external",
            {"external_services": ["Salesforce"]},
        ),
    ]


def _file(path: str, evidence: list[FeatureEvidence] | None = None) -> FeatureFile:
    return FeatureFile(
        path=path,
        primary=True,
        reason="route" if evidence else "tables",
        evidence=evidence or [],
    )


def _projection_fixture():
    web_path = "web/apps/accounts/page.tsx"
    api_path = "backend/src/rox_core/api/accounts.py"
    webhook_path = "backend/src/rox_core/api/webhook.py"
    helper_path = "backend/src/rox_core/services/accounts.py"
    worker_path = "backend/src/rox_core/tasks/accounts.py"
    temporal_path = "backend/src/rox_core/workflows/accounts.py"
    task_types_path = "backend/src/tasks/types.py"
    route_tag = "accounts"
    files = [
        _file(
            web_path,
            [FeatureEvidence(kind="route", path=web_path, line=5, tag=route_tag)],
        ),
        _file(
            api_path,
            [FeatureEvidence(kind="route", path=api_path, line=10, tag=route_tag)],
        ),
        _file(webhook_path),
        _file(helper_path),
        _file(worker_path),
        _file(temporal_path),
        _file("backend/src/unreached.py"),
    ]
    facts = {
        web_path: FileFacts(
            path=web_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
        ),
        api_path: FileFacts(
            path=api_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts/list",
                    handler="list_accounts",
                    line=11,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        webhook_path: FileFacts(
            path=webhook_path,
            api_group="provider",
            endpoints=[
                Endpoint(
                    method="POST",
                    path="/webhook/accounts",
                    handler="receive_accounts",
                    line=12,
                    deploy_target="WEBHOOK",
                )
            ],
            workers=[],
            externals=[],
        ),
        helper_path: FileFacts(
            path=helper_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[
                ExternalCall(service="Salesforce", module="salesforce", line=21)
            ],
            task_producers=[
                TaskProducer(task_type="TASK_A", path=helper_path, line=20)
            ],
            workflow_starts=[22],
        ),
        worker_path: FileFacts(
            path=worker_path,
            api_group=None,
            endpoints=[],
            workers=[Worker(name="SyncExecutor", kind="task_executor", line=4)],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="TASK_A",
                    executor="SyncExecutor",
                    path=worker_path,
                    line=30,
                ),
                TaskConsumer(
                    task_type="TASK_B",
                    executor="SyncExecutor",
                    path=worker_path,
                    line=31,
                ),
            ],
        ),
        temporal_path: FileFacts(
            path=temporal_path,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(
                    name="AccountsWorkflow",
                    kind="temporal_workflow",
                    line=7,
                )
            ],
            externals=[],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="TASK_A",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=40,
                ),
                TaskType(
                    name="TASK_B",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=41,
                ),
            ],
        ),
    }
    imports = {api_path: {helper_path: 19}}
    table_accesses = {helper_path: {"accounts": (23, 24)}}
    projection = project_component_diagram(
        files,
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports=imports,
        table_accesses=table_accesses,
        scope_tables={"accounts"},
        external_callers={},
    )
    return projection, helper_path


def test_flow_notes_group_by_column_and_move_queue_poll_to_worker() -> None:
    catalog = [
        _entry("http_clients", "HTTP clients", "client", "Callers", "component", {}),
        _entry(
            "interaction",
            "INTERACTION (Flask)",
            "service",
            "HTTP services",
            "component",
            {},
        ),
        _entry(
            "temporal",
            "Temporal workflows",
            "queue",
            "Workflows",
            "component",
            {},
        ),
        _entry("sqs", "SQS queues", "queue", "Queues", "queue", {}),
        _entry(
            "data_workers",
            "INTEGRATION workers",
            "service(many)",
            "Background workers",
            "component",
            {},
        ),
        _entry("postgres", "PostgreSQL", "store", "Stores", "store", {}),
        _entry(
            "crm", "CRM + workspace APIs", "external", "Provider APIs", "external", {}
        ),
    ]
    catalog_by_id = {entry["id"]: entry for entry in catalog}
    source_by_line = {
        line: CodeSource(path="runtime.py", lines=(line, line)) for line in range(1, 11)
    }
    nodes = [
        Node(
            id=entry["id"],
            label=entry["label"],
            source=source_by_line[index + 1],
            group=f"column-{entry['column'].lower().replace(' ', '-')}",
        )
        for index, entry in enumerate(catalog)
    ]
    edges = [
        Edge(
            src="http_clients",
            dst="interaction",
            label="REST calls",
            source=source_by_line[1],
        ),
        Edge(
            src="interaction",
            dst="postgres",
            label="reads + writes",
            source=source_by_line[2],
        ),
        Edge(
            src="interaction",
            dst="sqs",
            label="enqueue tasks",
            source=source_by_line[3],
        ),
        Edge(
            src="interaction",
            dst="temporal",
            label="start workflow",
            source=source_by_line[4],
        ),
        Edge(
            src="interaction",
            dst="crm",
            label="API calls",
            source=source_by_line[5],
        ),
        Edge(
            src="sqs",
            dst="data_workers",
            label="long-poll",
            source=source_by_line[6],
        ),
        Edge(
            src="data_workers",
            dst="postgres",
            label="reads + writes",
            source=source_by_line[7],
        ),
        Edge(
            src="temporal",
            dst="sqs",
            label="enqueue tasks",
            source=source_by_line[8],
        ),
    ]
    backward_edges = [
        Edge(
            src="data_workers",
            dst="temporal",
            label="start workflow",
            source=source_by_line[9],
        ),
        Edge(
            src="data_workers",
            dst="sqs",
            label="enqueue tasks",
            source=source_by_line[10],
        ),
    ]

    notes = _flow_notes(
        nodes,
        edges,
        catalog_by_id,
        backward_edges=backward_edges,
    )

    assert [note.text for note in notes] == [
        "HTTP clients make REST calls to INTERACTION (Flask).",
        "INTERACTION (Flask) reads + writes PostgreSQL, enqueues SQS tasks, starts "
        "Temporal workflows and makes API calls to CRM + workspace APIs.",
        "Temporal workflows enqueue SQS tasks.",
        "INTEGRATION workers long-poll SQS and read + write PostgreSQL, and also "
        "enqueue follow-up SQS tasks and start Temporal workflows.",
    ]
    assert {source.lines for source in notes[1].sources} == {
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    }
    assert {source.lines for source in notes[3].sources} == {
        (6, 6),
        (7, 7),
        (9, 9),
        (10, 10),
    }
    assert not any("SQS queues long-poll" in note.text for note in notes)


def test_flow_notes_group_boxes_by_column_and_keep_all_sources() -> None:
    catalog = [
        _entry(
            f"service_{index}",
            f"Service {index}",
            "service",
            "HTTP services",
            "component",
            {},
        )
        for index in range(7)
    ]
    catalog.append(_entry("postgres", "PostgreSQL", "store", "Stores", "store", {}))
    catalog_by_id = {entry["id"]: entry for entry in catalog}
    nodes = [
        Node(
            id=entry["id"],
            label=entry["label"],
            source=CodeSource(path="services.py", lines=(1, 1)),
            group=f"column-{entry['column'].lower().replace(' ', '-')}",
        )
        for entry in catalog
    ]
    edges = [
        Edge(
            src=f"service_{index}",
            dst="postgres",
            label="reads",
            source=CodeSource(path="services.py", lines=(index + 1, index + 1)),
        )
        for index in range(7)
    ]

    notes = _flow_notes(nodes, edges, catalog_by_id)

    assert len(notes) == 1
    assert notes[0].text == "HTTP services read PostgreSQL."
    assert {source.lines for source in notes[0].sources} == {
        (index + 1, index + 1) for index in range(7)
    }


def test_flow_notes_cap_each_edge_to_the_first_three_sources() -> None:
    catalog = [
        _entry(
            "interaction",
            "INTERACTION (Flask)",
            "service",
            "HTTP services",
            "component",
            {},
        ),
        _entry("postgres", "PostgreSQL", "store", "Stores", "store", {}),
    ]
    catalog_by_id = {entry["id"]: entry for entry in catalog}
    sources = [
        CodeSource(path="backend/src/notes.py", lines=(line, line))
        for line in (50, 10, 40, 20, 30)
    ]
    nodes = [
        Node(
            id=entry["id"],
            label=entry["label"],
            source=CodeSource(path="catalog.py", lines=(1, 1)),
            group=f"column-{entry['column'].lower().replace(' ', '-')}",
        )
        for entry in catalog
    ]
    edges = [
        Edge(
            src="interaction",
            dst="postgres",
            label="reads",
            source=sources[0],
        )
    ]

    notes = _flow_notes(
        nodes,
        edges,
        catalog_by_id,
        sources_by_pair={("interaction", "postgres"): list(reversed(sources))},
    )

    assert notes[0].sources == [
        CodeSource(path="backend/src/notes.py", lines=(10, 10)),
        CodeSource(path="backend/src/notes.py", lines=(20, 20)),
        CodeSource(path="backend/src/notes.py", lines=(30, 30)),
    ]


def test_component_projection_attributes_helpers_and_cites_runtime_flow() -> None:
    projection, helper_path = _projection_fixture()
    diagram = projection.diagram

    assert {node.id for node in diagram.nodes} == {
        "web",
        "provider_push",
        "interaction",
        "webhook",
        "temporal",
        "sqs",
        "data_workers",
        "postgres",
        "crm",
    }
    assert {(edge.src, edge.dst, edge.label) for edge in diagram.edges} == {
        ("web", "interaction", "REST calls"),
        ("provider_push", "webhook", "provider events"),
        ("interaction", "postgres", "reads + writes"),
        ("interaction", "sqs", "enqueue tasks"),
        ("interaction", "temporal", "start workflow"),
        ("interaction", "crm", "API calls"),
        ("sqs", "data_workers", "long-poll"),
    }
    assert not any(edge.src == "http_clients" for edge in diagram.edges)
    for edge in diagram.edges:
        assert edge.source.lines
    for group in diagram.groups:
        assert group.source.lines
    for node in diagram.nodes:
        assert node.source.lines
        for detail in node.details:
            assert detail.sources
    helper_edges = {
        edge.label: edge for edge in diagram.edges if edge.source.path == helper_path
    }
    assert {
        "reads + writes",
        "enqueue tasks",
        "start workflow",
        "API calls",
    } <= helper_edges.keys()
    details_by_node = {
        node.id: {detail.text for detail in node.details} for node in diagram.nodes
    }
    assert "/accounts: 1 endpoints" in details_by_node["interaction"]
    assert "INTEGRATION · 2 task types" in details_by_node["data_workers"]
    assert "TASK_A" not in " ".join(details_by_node["data_workers"])
    assert "TASK_B" not in " ".join(details_by_node["data_workers"])
    assert "AccountsWorkflow" in details_by_node["temporal"]
    assert "IntegrationQueueType · 2 task types" in details_by_node["sqs"]
    assert "TASK_A" not in " ".join(details_by_node["sqs"])
    assert "TASK_B" not in " ".join(details_by_node["sqs"])
    assert "accounts" in details_by_node["postgres"]
    assert "Salesforce" in details_by_node["crm"]
    assert any("CRM + workspace APIs" in note.text for note in diagram.notes)
    assert not any("Salesforce" in note.text for note in diagram.notes)
    assert diagram.unreached_files == ["backend/src/unreached.py"]
    assert 3 <= len(diagram.notes) <= 6
    assert all(note.sources for note in diagram.notes)


def test_provider_box_merging_keeps_component_count_within_budget() -> None:
    api_path = "backend/src/api/accounts.py"
    helper_path = "backend/src/services/accounts.py"
    providers = [
        _entry(
            f"provider_{index:02}",
            f"Provider {index:02}",
            "external",
            "Provider APIs",
            "external",
            {"external_services": [f"Service{index:02}"]},
        )
        for index in range(15)
    ]
    catalog = [
        _entry(
            "interaction",
            "INTERACTION (Flask)",
            "service",
            "HTTP services",
            "component",
            {"deploy_targets": ["INTERACTION"]},
        ),
        *providers,
    ]
    facts = {
        api_path: FileFacts(
            path=api_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="list_accounts",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        helper_path: FileFacts(
            path=helper_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[
                ExternalCall(
                    service=f"Service{index:02}",
                    module=f"service_{index:02}",
                    line=index + 1,
                )
                for index in range(15)
            ],
        ),
    }
    projection = project_component_diagram(
        [_file(api_path), _file(helper_path)],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={api_path: {helper_path: 4}},
        table_accesses={},
        scope_tables=set(),
        external_callers={},
    )

    assert len(projection.diagram.nodes) <= 15
    assert {node.id for node in projection.diagram.nodes} == {
        "interaction",
        "provider-apis",
    }
    assert len(projection.diagram.edges) == 1
    assert projection.diagram.edges[0].src == "interaction"
    assert projection.diagram.edges[0].dst == "provider-apis"
    assert projection.diagram.edges[0].label == "API calls"


def test_shared_helper_is_owned_by_nearest_entry_component() -> None:
    near_path = "backend/src/rox_core/api/accounts.py"
    far_path = "backend/src/rox_core/api/webhook.py"
    bridge_path = "backend/src/rox_core/services/bridge.py"
    helper_path = "backend/src/rox_core/services/shared.py"
    facts = {
        near_path: FileFacts(
            path=near_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="list_accounts",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        far_path: FileFacts(
            path=far_path,
            api_group="webhook",
            endpoints=[
                Endpoint(
                    method="POST",
                    path="/webhook",
                    handler="receive_webhook",
                    line=4,
                    deploy_target="WEBHOOK",
                )
            ],
            workers=[],
            externals=[],
        ),
        bridge_path: FileFacts(
            path=bridge_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
        ),
        helper_path: FileFacts(
            path=helper_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
    }
    projection = project_component_diagram(
        [_file(path) for path in (near_path, far_path, bridge_path, helper_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={
            near_path: {helper_path: 5},
            far_path: {bridge_path: 6},
            bridge_path: {helper_path: 7},
        },
        table_accesses={},
        scope_tables=set(),
        external_callers={},
    )

    provider_edges = [
        edge
        for edge in projection.diagram.edges
        if edge.dst == "crm" and edge.label == "API calls"
    ]
    assert len(provider_edges) == 1
    assert provider_edges[0].src == "interaction"
    assert provider_edges[0].source.path == helper_path


def test_external_sibling_caller_qualifies_and_seeds_its_scope_file() -> None:
    target_path = "backend/src/rox_core/api/integrations/business.py"
    caller_path = "backend/src/rox_core/api/integrations/routes.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
        caller_path: FileFacts(
            path=caller_path,
            api_group="integrations",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/integrations",
                    handler="get_integrations",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
    }
    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={caller_path: [target_path]},
    )

    assert {(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges} == {
        ("http_clients", "interaction", "REST calls"),
        ("interaction", "crm", "API calls"),
    }
    assert {node.id for node in projection.diagram.nodes} == {
        "http_clients",
        "interaction",
        "crm",
    }
    assert (
        next(edge for edge in projection.diagram.edges if edge.dst == "crm").source.path
        == target_path
    )
    assert projection.diagram.unreached_files == []


def test_external_worker_caller_outside_sibling_directory_seeds_worker_pool() -> None:
    target_path = (
        "backend/src/ext_integrations/calendar_integration/calendar_event_service.py"
    )
    caller_path = "backend/src/tasks/executors/calendar_delete.py"
    endpoint_caller = "backend/src/chat/routes/calendar_events.py"
    task_types_path = "backend/src/tasks/task_types.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
        caller_path: FileFacts(
            path=caller_path,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(name="CalendarDeleteExecutor", kind="task_executor", line=9)
            ],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="CALENDAR_DELETE",
                    executor="CalendarDeleteExecutor",
                    path=caller_path,
                    line=28,
                )
            ],
        ),
        endpoint_caller: FileFacts(
            path=endpoint_caller,
            api_group="chat",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/calendar-events",
                    handler="list_calendar_events",
                    line=3,
                    deploy_target="WEBHOOK",
                )
            ],
            workers=[],
            externals=[],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="CALENDAR_DELETE",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=40,
                )
            ],
        ),
    }
    catalog = _runtime_catalog()
    catalog = [
        next(entry for entry in catalog if entry["id"] == "webhook"),
        *(entry for entry in catalog if entry["id"] != "webhook"),
    ]

    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={},
        table_accesses={target_path: {"calendar_event": (None, 8)}},
        scope_tables={"calendar_event"},
        external_callers={
            caller_path: [target_path],
            endpoint_caller: [target_path],
        },
        external_caller_hops={endpoint_caller: 1},
    )

    assert {(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges} == {
        ("data_workers", "crm", "API calls"),
        ("data_workers", "postgres", "reads"),
        ("sqs", "data_workers", "long-poll"),
    }
    assert target_path not in projection.diagram.unreached_files
    assert "webhook" not in {node.id for node in projection.diagram.nodes}


def test_in_scope_entry_ownership_precedes_external_endpoint_and_worker_seeds() -> None:
    entry_path = "backend/src/rox_core/api/accounts/routes.py"
    target_path = "backend/src/rox_core/services/accounts.py"
    sibling_endpoint = "backend/src/rox_core/api/accounts/admin_routes.py"
    worker_paths = [
        f"backend/src/tasks/executors/accounts_sync_{index}.py" for index in range(3)
    ]
    task_types_path = "backend/src/tasks/task_types.py"
    facts = {
        entry_path: FileFacts(
            path=entry_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="list_accounts",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
        sibling_endpoint: FileFacts(
            path=sibling_endpoint,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/admin/accounts",
                    handler="list_admin_accounts",
                    line=3,
                    deploy_target="WEBHOOK",
                )
            ],
            workers=[],
            externals=[],
        ),
    }
    task_types = []
    external_callers = {sibling_endpoint: [target_path]}
    for index, worker_path in enumerate(worker_paths):
        name = f"AccountsSyncExecutor{index}"
        task_name = f"ACCOUNTS_SYNC_{index}"
        facts[worker_path] = FileFacts(
            path=worker_path,
            api_group=None,
            endpoints=[],
            workers=[Worker(name=name, kind="task_executor", line=9)],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type=task_name,
                    executor=name,
                    path=worker_path,
                    line=28,
                )
            ],
        )
        task_types.append(
            TaskType(
                name=task_name,
                queue_type="IntegrationQueueType.DEFAULT",
                queue_class="IntegrationQueueType",
                deploy_target="INTEGRATION",
                path=task_types_path,
                line=40 + index,
            )
        )
        external_callers[worker_path] = [target_path]
    facts[task_types_path] = FileFacts(
        path=task_types_path,
        api_group=None,
        endpoints=[],
        workers=[],
        externals=[],
        task_types=task_types,
    )

    projection = project_component_diagram(
        [_file(entry_path), _file(target_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={entry_path: {target_path: 5}},
        table_accesses={},
        scope_tables=set(),
        external_callers=external_callers,
    )

    assert ("interaction", "crm", "API calls") in {
        (edge.src, edge.dst, edge.label) for edge in projection.diagram.edges
    }
    assert "data_workers" not in {node.id for node in projection.diagram.nodes}
    assert "webhook" not in {node.id for node in projection.diagram.nodes}


def test_worker_task_counts_and_long_poll_edges_use_only_owned_executors() -> None:
    target_path = "backend/src/feature/calendar/service.py"
    worker_a = "backend/src/tasks/executors/calendar_feature.py"
    worker_b = "backend/src/tasks/executors/unrelated.py"
    duplicate_worker = "backend/src/tasks/executors/unregistered_calendar.py"
    registry_path = "backend/src/tasks/executor_registry.py"
    task_types_path = "backend/src/tasks/task_types.py"
    catalog = _runtime_catalog()
    catalog.append(
        _entry(
            "agent_workers",
            "ASYNC_AGENT",
            "service(many)",
            "Background workers",
            "component",
            {
                "deploy_targets": ["ASYNC_AGENT"],
                "queue_type_classes": ["AsyncAgentQueueType"],
            },
        )
    )
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="UNSCOPED_TASK",
                    executor="UnownedExecutor",
                    path=target_path,
                    line=10,
                )
            ],
            task_producers=[
                TaskProducer(
                    task_type="UNSCOPED_TASK",
                    path=target_path,
                    line=11,
                )
            ],
        ),
        worker_a: FileFacts(
            path=worker_a,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(name="CalendarFeatureExecutor", kind="task_executor", line=3)
            ],
            externals=[],
        ),
        worker_b: FileFacts(
            path=worker_b,
            api_group=None,
            endpoints=[],
            workers=[Worker(name="UnrelatedExecutor", kind="task_executor", line=3)],
            externals=[],
        ),
        duplicate_worker: FileFacts(
            path=duplicate_worker,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(name="CalendarFeatureExecutor", kind="task_executor", line=3)
            ],
            externals=[],
        ),
        registry_path: FileFacts(
            path=registry_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="CALENDAR_FEATURE_TASK",
                    executor="CalendarFeatureExecutor",
                    path=registry_path,
                    line=7,
                ),
                TaskConsumer(
                    task_type="UNRELATED_TASK",
                    executor="UnrelatedExecutor",
                    path=registry_path,
                    line=8,
                ),
            ],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="CALENDAR_FEATURE_TASK",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=20,
                ),
                TaskType(
                    name="UNRELATED_TASK",
                    queue_type="AsyncAgentQueueType.DEFAULT",
                    queue_class="AsyncAgentQueueType",
                    deploy_target="ASYNC_AGENT",
                    path=task_types_path,
                    line=21,
                ),
                TaskType(
                    name="UNSCOPED_TASK",
                    queue_type="AsyncAgentQueueType.DEFAULT",
                    queue_class="AsyncAgentQueueType",
                    deploy_target="ASYNC_AGENT",
                    path=task_types_path,
                    line=22,
                ),
            ],
        ),
    }

    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={
            registry_path: {worker_a: 1, worker_b: 2},
        },
        table_accesses={},
        scope_tables=set(),
        external_callers={worker_a: [target_path]},
    )
    details = {
        node.id: {detail.text for detail in node.details}
        for node in projection.diagram.nodes
    }

    assert {(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges} == {
        ("sqs", "data_workers", "long-poll")
    }
    assert "INTEGRATION · 1 task types" in details["data_workers"]
    assert "IntegrationQueueType · 1 task types" in details["sqs"]
    assert not any(
        detail.startswith("AsyncAgentQueueType") for detail in details["sqs"]
    )
    assert "agent_workers" not in details
    assert all("ASYNC_AGENT" not in text for text in details["sqs"])


def test_nearest_worker_caller_owns_task_scope_before_distant_caller() -> None:
    target_path = "backend/src/feature/calendar/service.py"
    direct_worker = "backend/src/tasks/executors/calendar_direct.py"
    distant_worker = "backend/src/tasks/executors/calendar_distant.py"
    registry_path = "backend/src/tasks/executor_registry.py"
    task_types_path = "backend/src/tasks/task_types.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
        ),
        direct_worker: FileFacts(
            path=direct_worker,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(name="DirectCalendarExecutor", kind="task_executor", line=3)
            ],
            externals=[],
        ),
        distant_worker: FileFacts(
            path=distant_worker,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(name="DistantCalendarExecutor", kind="task_executor", line=3)
            ],
            externals=[],
        ),
        registry_path: FileFacts(
            path=registry_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="DIRECT_CALENDAR_TASK",
                    executor="DirectCalendarExecutor",
                    path=registry_path,
                    line=7,
                ),
                TaskConsumer(
                    task_type="DISTANT_CALENDAR_TASK",
                    executor="DistantCalendarExecutor",
                    path=registry_path,
                    line=8,
                ),
            ],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="DIRECT_CALENDAR_TASK",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=20,
                ),
                TaskType(
                    name="DISTANT_CALENDAR_TASK",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=21,
                ),
            ],
        ),
    }

    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={
            registry_path: {direct_worker: 1, distant_worker: 2},
        },
        table_accesses={},
        scope_tables=set(),
        external_callers={
            direct_worker: [target_path],
            distant_worker: [target_path],
        },
        external_caller_hops={direct_worker: 1, distant_worker: 2},
    )
    details = {
        node.id: {detail.text for detail in node.details}
        for node in projection.diagram.nodes
    }

    assert {(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges} == {
        ("sqs", "data_workers", "long-poll")
    }
    assert "INTEGRATION · 1 task types" in details["data_workers"]
    assert "IntegrationQueueType · 1 task types" in details["sqs"]


def test_endpoints_and_executors_in_one_file_map_to_service_and_pool() -> None:
    path = "backend/src/rox_core/api/integrations/mixed.py"
    task_types_path = "backend/src/tasks/task_types.py"
    catalog = _runtime_catalog()
    data_workers = next(entry for entry in catalog if entry["id"] == "data_workers")
    interaction = next(entry for entry in catalog if entry["id"] == "interaction")
    interaction["match"]["path_prefixes"] = ["backend/src/rox_core/api/"]
    sqs = next(entry for entry in catalog if entry["id"] == "sqs")
    catalog = [
        *catalog[:3],
        sqs,
        data_workers,
        *(entry for entry in catalog[3:] if entry["id"] not in {"data_workers", "sqs"}),
    ]
    facts = {
        path: FileFacts(
            path=path,
            api_group="integrations",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/mixed",
                    handler="get_mixed",
                    line=4,
                    deploy_target="INTEGRATION",
                )
            ],
            workers=[Worker(name="MixedExecutor", kind="task_executor", line=10)],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="MIXED_TASK",
                    executor="MixedExecutor",
                    path=path,
                    line=12,
                )
            ],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="MIXED_TASK",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=20,
                )
            ],
        ),
    }

    projection = project_component_diagram(
        [_file(path)],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={},
    )
    details = {
        node.id: {detail.text for detail in node.details}
        for node in projection.diagram.nodes
    }

    assert "/mixed: 1 endpoints" in details["interaction"]
    assert not any("endpoints" in text for text in details["data_workers"])
    assert "INTEGRATION · 1 task types" in details["data_workers"]
    assert ("sqs", "data_workers", "long-poll") in {
        (edge.src, edge.dst, edge.label) for edge in projection.diagram.edges
    }


def test_external_temporal_caller_outside_sibling_directory_seeds_temporal() -> None:
    target_path = (
        "backend/src/ext_integrations/calendar_integration/calendar_event_service.py"
    )
    caller_path = "backend/src/temporal/workflows/calendar_events.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
        ),
        caller_path: FileFacts(
            path=caller_path,
            api_group=None,
            endpoints=[],
            workers=[
                Worker(
                    name="CalendarEventsWorkflow",
                    kind="temporal_activity",
                    line=9,
                )
            ],
            externals=[],
        ),
    }

    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={},
        table_accesses={target_path: {"calendar_event": (None, 8)}},
        scope_tables={"calendar_event"},
        external_callers={caller_path: [target_path]},
    )

    assert [(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges] == [
        ("temporal", "postgres", "reads")
    ]


def test_sibling_endpoint_ownership_precedes_remote_endpoint_and_worker_seed() -> None:
    target_path = "backend/src/rox_core/api/integrations/business.py"
    sibling_caller = "backend/src/rox_core/api/integrations/routes.py"
    remote_caller = "backend/src/chat/routes/accounts.py"
    worker_caller = "backend/src/tasks/executors/accounts_sync.py"
    task_types_path = "backend/src/tasks/task_types.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
        sibling_caller: FileFacts(
            path=sibling_caller,
            api_group="integrations",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/integrations",
                    handler="get_integrations",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        remote_caller: FileFacts(
            path=remote_caller,
            api_group="chat",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="get_accounts",
                    line=3,
                    deploy_target="WEBHOOK",
                )
            ],
            workers=[],
            externals=[],
        ),
        worker_caller: FileFacts(
            path=worker_caller,
            api_group=None,
            endpoints=[],
            workers=[Worker(name="AccountsSyncExecutor", kind="task_executor", line=9)],
            externals=[],
            task_consumers=[
                TaskConsumer(
                    task_type="ACCOUNTS_SYNC",
                    executor="AccountsSyncExecutor",
                    path=worker_caller,
                    line=28,
                )
            ],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="ACCOUNTS_SYNC",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=40,
                )
            ],
        ),
    }
    catalog = _runtime_catalog()
    catalog = [
        next(entry for entry in catalog if entry["id"] == "webhook"),
        *(entry for entry in catalog if entry["id"] != "webhook"),
    ]

    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={
            sibling_caller: [target_path],
            remote_caller: [target_path],
            worker_caller: [target_path],
        },
        external_caller_hops={sibling_caller: 3, remote_caller: 1},
    )

    assert [(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges] == [
        ("interaction", "crm", "API calls")
    ]
    assert "webhook" not in {node.id for node in projection.diagram.nodes}


def test_endpoint_fallback_uses_only_minimum_hop_caller_when_unseeded() -> None:
    target_path = "backend/src/rox_core/api/integrations/business.py"
    nearest_caller = "backend/src/chat/routes/calendar_events.py"
    distant_caller = "backend/src/rox_core/api/admin/routes.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
        nearest_caller: FileFacts(
            path=nearest_caller,
            api_group="chat",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="get_accounts",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        distant_caller: FileFacts(
            path=distant_caller,
            api_group="admin",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/admin/integrations",
                    handler="get_integrations",
                    line=4,
                    deploy_target="WEBHOOK",
                )
            ],
            workers=[],
            externals=[],
        ),
    }
    catalog = _runtime_catalog()
    catalog = [
        next(entry for entry in catalog if entry["id"] == "webhook"),
        *(entry for entry in catalog if entry["id"] != "webhook"),
    ]
    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={
            nearest_caller: [target_path],
            distant_caller: [target_path],
        },
        external_caller_hops={nearest_caller: 1, distant_caller: 3},
    )

    assert [(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges] == [
        ("interaction", "crm", "API calls")
    ]
    assert "webhook" not in {node.id for node in projection.diagram.nodes}
    assert projection.diagram.unreached_files == []


def test_ineligible_nearest_seed_falls_back_to_nearest_eligible_owner() -> None:
    entry_path = "backend/src/rox_core/api/accounts.py"
    bridge_paths = [
        "backend/src/rox_core/services/bridge_one.py",
        "backend/src/rox_core/services/bridge_two.py",
    ]
    helper_path = "backend/src/rox_core/services/shared.py"
    external_target = "backend/src/rox_core/api/external_accounts.py"
    caller_path = "backend/src/chat/routes/accounts.py"
    catalog = [
        *_runtime_catalog(),
        _entry(
            "chat",
            "CHAT (FastAPI)",
            "service",
            "HTTP services",
            "component",
            {"deploy_targets": ["CHAT"]},
        ),
    ]
    facts = {
        entry_path: FileFacts(
            path=entry_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="get_accounts",
                    line=3,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
        **{
            path: FileFacts(
                path=path,
                api_group=None,
                endpoints=[],
                workers=[],
                externals=[],
            )
            for path in [*bridge_paths, external_target]
        },
        helper_path: FileFacts(
            path=helper_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=8)],
        ),
        caller_path: FileFacts(
            path=caller_path,
            api_group="chat",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="get_accounts",
                    line=3,
                    deploy_target="CHAT",
                )
            ],
            workers=[],
            externals=[],
        ),
    }
    projection = project_component_diagram(
        [
            _file(path)
            for path in [entry_path, *bridge_paths, helper_path, external_target]
        ],
        component_catalog=catalog,
        component_facts=facts,
        component_imports={
            entry_path: {bridge_paths[0]: 5},
            bridge_paths[0]: {bridge_paths[1]: 6},
            bridge_paths[1]: {helper_path: 7},
            external_target: {helper_path: 8},
        },
        table_accesses={},
        scope_tables=set(),
        external_callers={caller_path: [external_target]},
    )

    provider_edge = next(edge for edge in projection.diagram.edges if edge.dst == "crm")
    assert provider_edge.src == "interaction"
    assert provider_edge.source.path == helper_path
    assert "chat" not in {node.id for node in projection.diagram.nodes}
    assert projection.diagram.unreached_files == [external_target]


def test_backward_edges_become_cited_notes_and_keep_queue_consumer_edge() -> None:
    worker_path = "backend/src/tasks/pool.py"
    api_path = "backend/src/rox_core/api/accounts.py"
    task_types_path = "backend/src/tasks/task_types.py"
    facts = {
        worker_path: FileFacts(
            path=worker_path,
            api_group=None,
            endpoints=[],
            workers=[Worker(name="PoolExecutor", kind="task_executor", line=8)],
            externals=[],
            task_producers=[
                TaskProducer(task_type="TASK_A", path=worker_path, line=20)
            ],
            workflow_starts=[21],
            task_consumers=[
                TaskConsumer(
                    task_type="TASK_A",
                    executor="PoolExecutor",
                    path=worker_path,
                    line=22,
                )
            ],
        ),
        api_path: FileFacts(
            path=api_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="list_accounts",
                    line=4,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[ExternalCall(service="Salesforce", module="salesforce", line=5)],
        ),
        task_types_path: FileFacts(
            path=task_types_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
            task_types=[
                TaskType(
                    name="TASK_A",
                    queue_type="IntegrationQueueType.DEFAULT",
                    queue_class="IntegrationQueueType",
                    deploy_target="INTEGRATION",
                    path=task_types_path,
                    line=15,
                )
            ],
        ),
    }
    projection = project_component_diagram(
        [_file(worker_path), _file(api_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={},
    )

    assert {(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges} == {
        ("http_clients", "interaction", "REST calls"),
        ("interaction", "crm", "API calls"),
        ("sqs", "data_workers", "long-poll"),
    }
    assert "data_workers" in {node.id for node in projection.diagram.nodes}
    note = next(
        (
            note
            for note in projection.diagram.notes
            if "INTEGRATION workers long-poll SQS" in note.text
        ),
        None,
    )
    assert note is not None, [claim.text for claim in projection.diagram.notes]
    assert 3 <= len(projection.diagram.notes) <= 6
    assert (
        note.text
        == "INTEGRATION workers long-poll SQS, and also enqueue follow-up SQS tasks and start Temporal workflows."
    )
    assert {source.lines for source in note.sources} == {
        (20, 20),
        (21, 21),
        (22, 22),
    }


def test_http_clients_edge_uses_first_service_endpoint_without_route_evidence() -> None:
    api_path = "backend/src/rox_core/api/accounts.py"
    facts = {
        api_path: FileFacts(
            path=api_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="POST",
                    path="/accounts/new",
                    handler="create_account",
                    line=30,
                    deploy_target="INTERACTION",
                ),
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="list_accounts",
                    line=10,
                    deploy_target="INTERACTION",
                ),
            ],
            workers=[],
            externals=[],
        )
    }
    projection = project_component_diagram(
        [_file(api_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={},
    )

    assert len(projection.diagram.edges) == 1
    edge = projection.diagram.edges[0]
    assert (edge.src, edge.dst, edge.label) == (
        "http_clients",
        "interaction",
        "REST calls",
    )
    assert edge.source.path == api_path
    assert edge.source.lines == (10, 10)
    assert any(
        note.text == "HTTP clients make REST calls to INTERACTION (Flask)."
        for note in projection.diagram.notes
    )


def test_external_endpoint_caller_details_belong_to_service_component() -> None:
    target_path = "backend/src/rox_core/services/accounts.py"
    caller_path = "backend/src/rox_core/api/accounts/routes.py"
    facts = {
        target_path: FileFacts(
            path=target_path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
        ),
        caller_path: FileFacts(
            path=caller_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="POST",
                    path="/msteams_agent",
                    handler="receive_msteams_agent",
                    line=18,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
        ),
    }

    projection = project_component_diagram(
        [_file(target_path)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={caller_path: [target_path]},
        external_caller_hops={caller_path: 1},
    )
    details = {
        node.id: {detail.text for detail in node.details}
        for node in projection.diagram.nodes
    }

    assert (
        "http_clients",
        "interaction",
        "REST calls",
    ) in {(edge.src, edge.dst, edge.label) for edge in projection.diagram.edges}
    assert "/msteams_agent: 1 endpoints" in details["interaction"]
    assert "data_workers" not in details


def test_temporal_workflow_details_show_two_names_then_remainder() -> None:
    api_path = "backend/src/rox_core/api/accounts.py"
    workflow_paths = [
        f"backend/src/rox_core/workflows/workflow_{index}.py" for index in range(4)
    ]
    facts = {
        api_path: FileFacts(
            path=api_path,
            api_group="accounts",
            endpoints=[
                Endpoint(
                    method="GET",
                    path="/accounts",
                    handler="list_accounts",
                    line=6,
                    deploy_target="INTERACTION",
                )
            ],
            workers=[],
            externals=[],
            workflow_starts=[8],
        ),
        **{
            path: FileFacts(
                path=path,
                api_group=None,
                endpoints=[],
                workers=[
                    Worker(
                        name=f"Workflow{index}",
                        kind="temporal_workflow",
                        line=index + 1,
                    )
                ],
                externals=[],
            )
            for index, path in enumerate(workflow_paths)
        },
    }
    projection = project_component_diagram(
        [_file(api_path), *(_file(path) for path in workflow_paths)],
        component_catalog=_runtime_catalog(),
        component_facts=facts,
        component_imports={},
        table_accesses={},
        scope_tables=set(),
        external_callers={},
    )

    temporal = next(node for node in projection.diagram.nodes if node.id == "temporal")
    assert [detail.text for detail in temporal.details] == [
        "Workflow0",
        "Workflow1",
        "+2 more",
    ]
    assert temporal.details[-1].sources


def test_edge_deduplication_merges_access_modes_and_uses_first_source() -> None:
    first_source = CodeSource(path="a.py", lines=(4, 4))
    later_source = CodeSource(path="b.py", lines=(2, 2))
    edges = _deduplicate_edges(
        [
            Edge(src="service", dst="postgres", label="writes", source=later_source),
            Edge(src="service", dst="postgres", label="reads", source=first_source),
        ]
    )

    assert len(edges) == 1
    assert edges[0].label == "reads + writes"
    assert edges[0].source == first_source


def test_pruning_drops_early_provider_edge_redundant_with_worker_edge() -> None:
    catalog = {entry["id"]: entry for entry in _runtime_catalog()}
    interaction_edge = Edge(
        src="interaction",
        dst="crm",
        label="API calls",
        source=CodeSource(path="interaction.py", lines=(1, 1)),
    )
    worker_edge = Edge(
        src="data_workers",
        dst="crm",
        label="API calls",
        source=CodeSource(path="worker.py", lines=(2, 2)),
    )

    kept, pruned = _prune_early_edges(
        [interaction_edge, worker_edge],
        catalog,
        {("interaction", "crm"): 1, ("data_workers", "crm"): 1},
    )

    assert kept == [worker_edge]
    assert pruned == [interaction_edge]


def test_pruning_keeps_best_cited_early_store_edge_then_catalog_order() -> None:
    entries = [
        _entry(
            "interaction", "INTERACTION", "service", "HTTP services", "component", {}
        ),
        _entry("temporal", "Temporal", "queue", "Workflows", "component", {}),
        _entry("webhook", "WEBHOOK", "service", "HTTP services", "component", {}),
        _entry("postgres", "PostgreSQL", "store", "Stores", "store", {}),
    ]
    catalog = {entry["id"]: entry for entry in entries}
    edges = [
        Edge(
            src=component_id,
            dst="postgres",
            label="reads",
            source=CodeSource(path=f"{component_id}.py", lines=(1, 1)),
        )
        for component_id in ("interaction", "webhook", "temporal")
    ]

    kept, pruned = _prune_early_edges(
        edges,
        catalog,
        {
            ("interaction", "postgres"): 2,
            ("webhook", "postgres"): 4,
            ("temporal", "postgres"): 4,
        },
    )

    assert [(edge.src, edge.dst) for edge in kept] == [("temporal", "postgres")]
    assert {edge.src for edge in pruned} == {"interaction", "webhook"}


def test_pruning_drops_fewest_cited_early_skip_edge_to_meet_edge_budget() -> None:
    entries = [
        _entry("caller_a", "Caller A", "client", "Callers", "component", {}),
        _entry("caller_b", "Caller B", "client", "Callers", "component", {}),
        _entry("http_a", "HTTP A", "service", "HTTP services", "component", {}),
        _entry("http_b", "HTTP B", "service", "HTTP services", "component", {}),
        _entry("temporal", "Temporal", "queue", "Workflows", "component", {}),
        _entry("sqs", "SQS", "queue", "Queues", "queue", {}),
        _entry("crm", "CRM", "external", "Provider APIs", "external", {}),
    ]
    catalog = {entry["id"]: entry for entry in entries}
    pairs = [
        ("caller_a", "http_a"),
        ("caller_a", "http_b"),
        ("caller_b", "http_b"),
        ("caller_a", "temporal"),
        ("caller_b", "temporal"),
        ("caller_a", "sqs"),
        ("caller_b", "sqs"),
        ("caller_a", "crm"),
        ("caller_b", "crm"),
        ("http_a", "crm"),
        ("http_b", "crm"),
        ("temporal", "sqs"),
    ]
    edges = [
        Edge(
            src=src,
            dst=dst,
            label="flow",
            source=CodeSource(path=f"{src}.py", lines=(index + 1, index + 1)),
        )
        for index, (src, dst) in enumerate(pairs)
    ]
    source_counts = {pair: 2 for pair in pairs}
    source_counts[("caller_b", "temporal")] = 1
    source_counts[("caller_b", "sqs")] = 3

    kept, pruned = _prune_early_edges(edges, catalog, source_counts)

    assert len(kept) == 11
    assert [(edge.src, edge.dst) for edge in pruned] == [("caller_b", "temporal")]


def test_flow_notes_include_pruned_edge_facts_and_all_citations() -> None:
    interaction = _entry(
        "interaction",
        "INTERACTION (Flask)",
        "service",
        "HTTP services",
        "component",
        {},
    )
    webhook = _entry(
        "webhook", "WEBHOOK (Flask)", "service", "HTTP services", "component", {}
    )
    postgres = _entry("postgres", "PostgreSQL", "store", "Stores", "store", {})
    catalog = {entry["id"]: entry for entry in (interaction, webhook, postgres)}
    drawn_source = CodeSource(path="webhook.py", lines=(5, 5))
    pruned_source = CodeSource(path="interaction.py", lines=(2, 2))
    additional_source = CodeSource(path="interaction.py", lines=(3, 3))
    nodes = [
        Node(
            id=component_id,
            label=catalog[component_id]["label"],
            source=CodeSource(path=f"{component_id}.py", lines=(1, 1)),
            group="column-http-services",
        )
        for component_id in ("interaction", "webhook")
    ]
    drawn = Edge(
        src="webhook",
        dst="postgres",
        label="reads",
        source=drawn_source,
    )
    pruned = Edge(
        src="interaction",
        dst="postgres",
        label="reads",
        source=pruned_source,
    )

    notes = _flow_notes(
        nodes,
        [drawn],
        catalog,
        pruned_edges=[pruned],
        sources_by_pair={
            ("interaction", "postgres"): [pruned_source, additional_source],
            ("webhook", "postgres"): [drawn_source],
        },
    )

    assert [note.text for note in notes] == ["HTTP services read PostgreSQL."]
    assert {source.path for source in notes[0].sources} == {
        "interaction.py",
        "webhook.py",
    }
    assert {source.lines[0] for source in notes[0].sources} == {2, 3, 5}


def test_flow_notes_show_common_actions_and_per_box_extras() -> None:
    entries = [
        _entry(
            "interaction",
            "INTERACTION (Flask)",
            "service",
            "HTTP services",
            "component",
            {},
        ),
        _entry(
            "webhook",
            "WEBHOOK (Flask)",
            "service",
            "HTTP services",
            "component",
            {},
        ),
        _entry("postgres", "PostgreSQL", "store", "Stores", "store", {}),
        _entry("sqs", "SQS queues", "queue", "Queues", "queue", {}),
    ]
    catalog = {entry["id"]: entry for entry in entries}
    nodes = [
        Node(
            id=component_id,
            label=catalog[component_id]["label"],
            source=CodeSource(path=f"{component_id}.py", lines=(1, 1)),
            group="column-http-services",
        )
        for component_id in ("webhook", "interaction")
    ]
    edges = [
        Edge(
            src=component_id,
            dst="postgres",
            label="reads",
            source=CodeSource(path=f"{component_id}.py", lines=(2, 2)),
        )
        for component_id in ("interaction", "webhook")
    ]
    edges.append(
        Edge(
            src="webhook",
            dst="sqs",
            label="enqueue tasks",
            source=CodeSource(path="webhook.py", lines=(3, 3)),
        )
    )

    notes = _flow_notes(nodes, edges, catalog)

    assert [note.text for note in notes] == [
        "WEBHOOK (Flask) and INTERACTION (Flask) read PostgreSQL; "
        "WEBHOOK (Flask) also enqueues SQS tasks."
    ]


def test_flow_notes_append_shared_backward_actions_to_common_worker_claim() -> None:
    entries = [
        _entry(
            "agent_workers",
            "AGENT",
            "service(many)",
            "Background workers",
            "component",
            {},
        ),
        _entry(
            "data_workers",
            "SOR · BATCH",
            "service(many)",
            "Background workers",
            "component",
            {},
        ),
        _entry("sqs", "SQS queues", "queue", "Queues", "queue", {}),
        _entry("postgres", "PostgreSQL", "store", "Stores", "store", {}),
        _entry(
            "crm",
            "CRM + workspace APIs",
            "external",
            "Provider APIs",
            "external",
            {},
        ),
        _entry(
            "temporal",
            "Temporal workflows",
            "queue",
            "Workflows",
            "component",
            {},
        ),
    ]
    catalog = {entry["id"]: entry for entry in entries}
    nodes = [
        Node(
            id=entry["id"],
            label=entry["label"],
            source=CodeSource(path=f"{entry['id']}.py", lines=(1, 1)),
            group=f"column-{entry['column'].lower().replace(' ', '-')}",
        )
        for entry in entries
    ]
    edges = []
    line = 2
    for worker_id in ("agent_workers", "data_workers"):
        for target_id, label in (
            ("sqs", "long-poll"),
            ("postgres", "reads + writes"),
            ("crm", "API calls"),
        ):
            edges.append(
                Edge(
                    src=worker_id if target_id != "sqs" else "sqs",
                    dst=target_id if target_id != "sqs" else worker_id,
                    label=label,
                    source=CodeSource(
                        path=f"{worker_id}.py",
                        lines=(line, line),
                    ),
                )
            )
            line += 1
    backward_edges = [
        Edge(
            src=worker_id,
            dst="sqs",
            label="enqueue tasks",
            source=CodeSource(
                path=f"{worker_id}.py",
                lines=(line, line),
            ),
        )
        for worker_id in ("agent_workers", "data_workers")
    ]
    backward_edges.append(
        Edge(
            src="data_workers",
            dst="temporal",
            label="start workflow",
            source=CodeSource(path="data_workers.py", lines=(line + 1, line + 1)),
        )
    )

    notes = _flow_notes(
        nodes,
        edges,
        catalog,
        backward_edges=backward_edges,
    )

    assert [note.text for note in notes] == [
        "AGENT workers and SOR · BATCH workers long-poll SQS, read + write "
        "PostgreSQL and make API calls to CRM + workspace APIs, and also enqueue "
        "follow-up SQS tasks; and also SOR · BATCH workers start Temporal workflows."
    ]
    assert len(notes[0].sources) == 9
