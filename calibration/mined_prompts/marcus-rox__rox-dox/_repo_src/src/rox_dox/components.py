from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from rox_dox.cache import get_or_compute
from rox_dox.features import (
    PARALLEL_FILE_CHUNKSIZE,
    PARALLEL_MIN_FILES,
    _git_blobs,
    _git_tree,
    parallel_map,
)
from rox_dox.model import Model


class Endpoint(Model):
    method: str
    path: str
    handler: str
    line: int
    deploy_target: str | None = None


class Worker(Model):
    name: str
    kind: Literal["task_executor", "temporal_workflow", "temporal_activity"]
    line: int


class ExternalCall(Model):
    service: str
    module: str
    line: int


class TaskType(Model):
    name: str
    queue_type: str
    queue_class: str | None = None
    deploy_target: str | None = None
    path: str
    line: int


class TaskConsumer(Model):
    task_type: str
    executor: str
    path: str
    line: int


class TaskProducer(Model):
    task_type: str
    path: str
    line: int


class TaskFacts(Model):
    task_types: list[TaskType]
    consumers: list[TaskConsumer]
    producers: list[TaskProducer]
    unmapped_queue_type_count: int = 0
    unmapped_queue_type_values: list[str] = Field(default_factory=list)


class FileFacts(Model):
    path: str
    api_group: str | None
    endpoints: list[Endpoint]
    workers: list[Worker]
    externals: list[ExternalCall]
    task_types: list[TaskType] = Field(default_factory=list)
    task_consumers: list[TaskConsumer] = Field(default_factory=list)
    task_producers: list[TaskProducer] = Field(default_factory=list)
    workflow_starts: list[int] = Field(default_factory=list)


def collect_task_facts(file_facts: Mapping[str, FileFacts]) -> TaskFacts:
    task_types = {
        (fact.name, fact.queue_type, fact.path, fact.line): fact
        for file_fact in file_facts.values()
        for fact in file_fact.task_types
    }
    consumers = {
        (fact.task_type, fact.executor, fact.path, fact.line): fact
        for file_fact in file_facts.values()
        for fact in file_fact.task_consumers
    }
    producers = {
        (fact.task_type, fact.path, fact.line): fact
        for file_fact in file_facts.values()
        for fact in file_fact.task_producers
    }
    ordered_task_types = sorted(
        task_types.values(),
        key=lambda fact: (
            fact.name,
            fact.queue_type or "",
            fact.path,
            fact.line,
        ),
    )
    unmapped = [fact for fact in ordered_task_types if fact.deploy_target is None]
    return TaskFacts(
        task_types=ordered_task_types,
        consumers=sorted(
            consumers.values(),
            key=lambda fact: (
                fact.task_type,
                fact.executor,
                fact.path,
                fact.line,
            ),
        ),
        producers=sorted(
            producers.values(),
            key=lambda fact: (fact.task_type, fact.path, fact.line),
        ),
        unmapped_queue_type_count=len(unmapped),
        unmapped_queue_type_values=sorted(
            {fact.queue_type for fact in unmapped},
        ),
    )


EXTERNAL_SERVICES = {
    "slack_sdk": "Slack",
    "twilio": "Twilio",
    "googleapiclient": "Google APIs",
    "google.auth": "Google APIs",
    "google.oauth2": "Google APIs",
    "google.cloud": "Google APIs",
    "google.api_core": "Google APIs",
    "simple_salesforce": "Salesforce",
    "hubspot": "HubSpot",
    "stripe": "Stripe",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "litellm": "LiteLLM",
    "boto3": "AWS",
    "msal": "Microsoft identity",
}
HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})
TASK_TYPE_CLASS = "TaskType"
TASK_TYPE_INFO_CALL = "TaskTypeInfo"
TASK_TYPE_KEYWORD = "task_type"
QUEUE_TYPE_KEYWORD = "queue_type"


def _expression_parts(expression: ast.expr) -> list[str] | None:
    if isinstance(expression, ast.Subscript):
        expression = expression.value
    parts = []
    while isinstance(expression, ast.Attribute):
        parts.append(expression.attr)
        expression = expression.value
    if not isinstance(expression, ast.Name):
        return None
    return [expression.id, *reversed(parts)]


def _call_name(expression: ast.expr) -> str | None:
    parts = _expression_parts(expression)
    return parts[-1] if parts else None


def _constant_string(expression: ast.expr | None) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    return None


def _keyword_string(call: ast.Call, name: str) -> str | None:
    return next(
        (
            value
            for keyword in call.keywords
            if keyword.arg == name
            and (value := _constant_string(keyword.value)) is not None
        ),
        None,
    )


def _assignment_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _assignment_names(item)]
    return []


def _assignments(body: list[ast.stmt]) -> list[tuple[str, ast.expr]]:
    assignments = []
    for statement in body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if value is None:
            continue
        assignments.extend(
            (name, value) for target in targets for name in _assignment_names(target)
        )
    return assignments


def _module_assignments(tree: ast.Module) -> list[tuple[str, ast.expr]]:
    return _assignments(tree.body)


def _namespace_names(tree: ast.Module) -> dict[str, str]:
    namespaces = {}
    for variable, value in _module_assignments(tree):
        if not isinstance(value, ast.Call):
            continue
        call_name = _call_name(value.func)
        if call_name is None or not call_name.endswith("Namespace"):
            continue
        namespace = _keyword_string(value, "name")
        if namespace is None and value.args:
            namespace = _constant_string(value.args[0])
        if namespace is not None:
            namespaces[variable] = namespace
    return namespaces


def _registered_paths(tree: ast.Module) -> dict[str, set[str]]:
    paths: defaultdict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or _call_name(node.func) != "add_namespace"
            or not node.args
            or not isinstance(node.args[0], ast.Name)
        ):
            continue
        path = _keyword_string(node, "path")
        if path is not None:
            paths[node.args[0].id].add(path)
    return dict(paths)


def _namespace_prefixes(repo: Path, commit: str) -> dict[str, frozenset[str]]:
    python_paths = [
        (path, object_id)
        for path, object_id in _git_tree(repo, commit)
        if path.startswith("backend/src/") and path.endswith(".py")
    ]
    sources = _git_blobs(repo, python_paths)
    registrations: defaultdict[str, set[str]] = defaultdict(set)
    for path in sorted(sources):
        source = sources[path]
        if "add_namespace" not in source:
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        for variable, paths in _registered_paths(tree).items():
            registrations[variable].update(paths)
    return {name: frozenset(paths) for name, paths in registrations.items()}


@lru_cache(maxsize=16)
def _cached_namespace_prefixes(repo: Path, commit: str) -> dict[str, frozenset[str]]:
    return _namespace_prefixes(repo, commit)


def _condition_values(expression: ast.expr, variable: str) -> set[str]:
    if isinstance(expression, ast.BoolOp):
        return set().union(
            *(_condition_values(value, variable) for value in expression.values)
        )
    if not isinstance(expression, ast.Compare) or len(expression.ops) != 1:
        return set()
    if not isinstance(expression.ops[0], ast.Eq) or len(expression.comparators) != 1:
        return set()
    left = _expression_parts(expression.left)
    right = _constant_string(expression.comparators[0])
    if left == [variable] and right is not None:
        return {right}
    right_parts = _expression_parts(expression.comparators[0])
    left_value = _constant_string(expression.left)
    if right_parts == [variable] and left_value is not None:
        return {left_value}
    return set()


def _namespace_registration_targets(
    tree: ast.Module,
) -> dict[str, frozenset[tuple[str, str]]]:
    registrations: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)

    def visit(node: ast.AST, target: str | None) -> None:
        if isinstance(node, ast.If):
            targets = _condition_values(node.test, "deploy_target")
            if targets:
                for selected in sorted(targets):
                    for statement in node.body:
                        visit(statement, selected)
                for statement in node.orelse:
                    visit(statement, target)
            else:
                for statement in (*node.body, *node.orelse):
                    visit(statement, target)
            return
        if isinstance(node, ast.Call) and _call_name(node.func) == "add_namespace":
            if target is not None and node.args and isinstance(node.args[0], ast.Name):
                prefix = _keyword_string(node, "path")
                if prefix is not None:
                    registrations[node.args[0].id].add((prefix, target))
        for child in ast.iter_child_nodes(node):
            visit(child, target)

    visit(tree, None)
    return {name: frozenset(values) for name, values in registrations.items()}


def _webhook_blueprint_paths(tree: ast.Module) -> frozenset[str]:
    imported_modules: dict[str, str] = {}
    registered_names: set[str] = set()

    def visit(node: ast.AST, webhook_branch: bool = False) -> None:
        if isinstance(node, ast.If):
            is_webhook = "WEBHOOK" in _condition_values(
                node.test, "deploy_target_for_blueprints"
            )
            for statement in node.body:
                visit(statement, webhook_branch or is_webhook)
            for statement in node.orelse:
                visit(statement, webhook_branch)
            return
        if webhook_branch and isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_modules[alias.asname or alias.name] = node.module
        if (
            webhook_branch
            and isinstance(node, ast.Call)
            and _call_name(node.func) == "register_blueprint"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            registered_names.add(node.args[0].id)
        for child in ast.iter_child_nodes(node):
            visit(child, webhook_branch)

    visit(tree)
    return frozenset(
        "backend/src/" + imported_modules[name].replace(".", "/") + ".py"
        for name in registered_names
        if name in imported_modules
    )


def _queue_deploy_targets(tree: ast.Module) -> dict[str, str]:
    classes_by_target: defaultdict[str, set[str]] = defaultdict(set)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any("QUEUE_CONFIGS" in _assignment_names(target) for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for deploy_target, config in zip(
            node.value.keys, node.value.values, strict=True
        ):
            target_name = _constant_string(deploy_target)
            if (
                target_name is None
                or not isinstance(config, (ast.Tuple, ast.List))
                or len(config.elts) < 2
            ):
                continue
            queue_class = _constant_string(config.elts[1])
            if queue_class is not None:
                classes_by_target[target_name].add(queue_class)
    targets_by_class: defaultdict[str, set[str]] = defaultdict(set)
    for deploy_target, queue_classes in classes_by_target.items():
        for queue_class in queue_classes:
            targets_by_class[queue_class].add(deploy_target)
    return {
        queue_class: next(iter(targets))
        for queue_class, targets in targets_by_class.items()
        if len(targets) == 1
    }


def _join_path(prefix: str, path: str) -> str:
    parts = [value.strip("/") for value in (prefix, path) if value.strip("/")]
    return "/" + "/".join(parts) if parts else "/"


def _decorator_call(decorator: ast.expr) -> tuple[list[str], ast.Call | None]:
    if isinstance(decorator, ast.Call):
        return _expression_parts(decorator.func) or [], decorator
    return _expression_parts(decorator) or [], None


def _flask_endpoints(
    tree: ast.Module,
    path: str,
    prefixes: dict[str, frozenset[str]],
    registrations: dict[str, frozenset[tuple[str, str]]],
) -> list[tuple[str, Endpoint]]:
    namespaces = _namespace_names(tree)
    endpoints = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            parts, call = _decorator_call(decorator)
            if len(parts) < 2 or parts[-1] != "route" or call is None:
                continue
            namespace_variable = parts[-2]
            namespace = namespaces.get(namespace_variable)
            if namespace is None:
                continue
            route_path = _constant_string(call.args[0]) if call.args else ""
            if route_path is None:
                continue
            registered = registrations.get(namespace_variable, frozenset())
            routed_targets = (
                sorted(registered)
                if registered
                else [
                    (
                        next(iter(prefixes[namespace_variable]))
                        if len(prefixes.get(namespace_variable, ())) == 1
                        else "/" + namespace.strip("/"),
                        "",
                    )
                ]
            )
            for method in node.body:
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name in HTTP_METHODS
                ):
                    for prefix, deploy_target in routed_targets:
                        endpoints.append(
                            (
                                namespace,
                                Endpoint(
                                    method=method.name.upper(),
                                    path=_join_path(prefix, route_path),
                                    handler=node.name,
                                    line=decorator.lineno,
                                    deploy_target=deploy_target or None,
                                ),
                            )
                        )
    return endpoints


def _blueprint_endpoints(
    tree: ast.Module,
    path: str,
    webhook_blueprint_paths: frozenset[str],
) -> list[tuple[str, Endpoint]]:
    if path not in webhook_blueprint_paths:
        return []
    blueprints = {}
    for variable, value in _module_assignments(tree):
        if not isinstance(value, ast.Call) or not (
            (_expression_parts(value.func) or [])[-1:] == ["Blueprint"]
        ):
            continue
        name = _constant_string(value.args[0]) if value.args else variable
        prefix = _keyword_string(value, "url_prefix") or ""
        blueprints[variable] = (name, prefix)
    endpoints = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            parts, call = _decorator_call(decorator)
            if len(parts) < 2 or parts[-1] != "route" or call is None:
                continue
            blueprint = blueprints.get(parts[-2])
            if blueprint is None:
                continue
            route_path = _constant_string(call.args[0]) if call.args else ""
            if route_path is None:
                continue
            methods = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "methods"
                ),
                None,
            )
            method_names = (
                [
                    value
                    for value in (_constant_string(item) for item in methods.elts)
                    if value is not None
                ]
                if isinstance(methods, (ast.List, ast.Tuple))
                else ["GET"]
            )
            group, prefix = blueprint
            for method in method_names:
                endpoints.append(
                    (
                        group,
                        Endpoint(
                            method=method.upper(),
                            path=_join_path(prefix, route_path),
                            handler=node.name,
                            line=decorator.lineno,
                            deploy_target="WEBHOOK",
                        ),
                    )
                )
    return endpoints


def _router_config(
    tree: ast.Module,
    path: str,
) -> dict[str, tuple[str, str]]:
    routers = {}
    for variable, value in _module_assignments(tree):
        if not isinstance(value, ast.Call):
            continue
        call_name = _call_name(value.func)
        if call_name != "APIRouter" and variable != "app":
            continue
        prefix = _keyword_string(value, "prefix")
        tags = next(
            (
                keyword.value
                for keyword in value.keywords
                if keyword.arg == "tags"
                and isinstance(keyword.value, (ast.List, ast.Tuple))
            ),
            None,
        )
        first_tag = (
            _constant_string(tags.elts[0])
            if isinstance(tags, (ast.List, ast.Tuple)) and tags.elts
            else None
        )
        group = (
            (prefix.strip("/") if prefix is not None else "")
            or first_tag
            or (PurePosixPath(path).stem)
        )
        routers[variable] = (prefix or "", group)
    return routers


def _fastapi_endpoints(
    tree: ast.Module,
    path: str,
) -> list[tuple[str, Endpoint]]:
    routers = _router_config(tree, path)
    routers.setdefault("app", ("", PurePosixPath(path).stem))
    endpoints = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            parts, call = _decorator_call(decorator)
            if len(parts) < 2 or parts[-1] not in HTTP_METHODS or call is None:
                continue
            router = routers.get(parts[-2])
            if router is None:
                continue
            route_path = _constant_string(call.args[0]) if call.args else ""
            if route_path is None:
                continue
            prefix, group = router
            endpoints.append(
                (
                    group,
                    Endpoint(
                        method=parts[-1].upper(),
                        path=_join_path(prefix, route_path),
                        handler=node.name,
                        line=decorator.lineno,
                    ),
                )
            )
    return endpoints


def _worker_facts(tree: ast.Module) -> list[Worker]:
    workers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("Base") and any(
                (parts := _expression_parts(base))
                and parts[-1].endswith("TaskExecutor")
                for base in node.bases
            ):
                workers.append(
                    Worker(
                        name=node.name,
                        kind="task_executor",
                        line=node.lineno,
                    )
                )
            if any(
                _decorator_call(decorator)[0][-2:] == ["workflow", "defn"]
                for decorator in node.decorator_list
            ):
                workers.append(
                    Worker(
                        name=node.name,
                        kind="temporal_workflow",
                        line=node.lineno,
                    )
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(
                _decorator_call(decorator)[0][-2:] == ["activity", "defn"]
                for decorator in node.decorator_list
            ):
                workers.append(
                    Worker(
                        name=node.name,
                        kind="temporal_activity",
                        line=node.lineno,
                    )
                )
    return workers


def _task_type_names(expression: ast.expr) -> list[str]:
    names = set()
    for node in ast.walk(expression):
        if not isinstance(node, ast.Attribute):
            continue
        parts = _expression_parts(node)
        if parts is not None and len(parts) >= 2 and parts[-2] == TASK_TYPE_CLASS:
            names.add(parts[-1])
    return sorted(names)


def _queue_type_class(expression: ast.expr | None) -> str | None:
    parts = _expression_parts(expression) if expression is not None else None
    return parts[0] if parts is not None and len(parts) == 2 else None


def _task_type_facts(
    tree: ast.Module,
    path: str,
    queue_deploy_targets: Mapping[str, str],
) -> list[TaskType]:
    facts = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != TASK_TYPE_CLASS:
            continue
        if not any(_call_name(base) == "Enum" for base in node.bases):
            continue
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign):
                targets, value = [statement.target], statement.value
            else:
                continue
            if (
                not isinstance(value, ast.Call)
                or _call_name(value.func) != TASK_TYPE_INFO_CALL
            ):
                continue
            queue_expression = next(
                (
                    keyword.value
                    for keyword in value.keywords
                    if keyword.arg == QUEUE_TYPE_KEYWORD
                ),
                None,
            )
            queue_type = (
                ast.unparse(queue_expression) if queue_expression is not None else None
            )
            if queue_type is None:
                continue
            queue_class = _queue_type_class(queue_expression)
            for target in targets:
                if isinstance(target, ast.Name):
                    facts.append(
                        TaskType(
                            name=target.id,
                            queue_type=queue_type,
                            queue_class=queue_class,
                            deploy_target=queue_deploy_targets.get(queue_class),
                            path=path,
                            line=statement.lineno,
                        )
                    )
    return sorted(
        facts,
        key=lambda fact: (
            fact.name,
            fact.queue_type or "",
            fact.path,
            fact.line,
        ),
    )


def _task_consumer_facts(tree: ast.Module, path: str) -> list[TaskConsumer]:
    consumers = []
    for _, value in _module_assignments(tree):
        if not isinstance(value, ast.Dict):
            continue
        for key, executor in zip(value.keys, value.values, strict=True):
            if not isinstance(key, ast.Attribute) or not isinstance(executor, ast.Name):
                continue
            parts = _expression_parts(key)
            if parts is None or len(parts) != 2 or parts[0] != TASK_TYPE_CLASS:
                continue
            consumers.append(
                TaskConsumer(
                    task_type=parts[1],
                    executor=executor.id,
                    path=path,
                    line=key.lineno,
                )
            )
    return sorted(
        consumers,
        key=lambda consumer: (
            consumer.task_type,
            consumer.executor,
            consumer.path,
            consumer.line,
        ),
    )


def _task_producer_facts(tree: ast.Module, path: str) -> list[TaskProducer]:
    producers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != TASK_TYPE_KEYWORD:
                continue
            producers.extend(
                TaskProducer(task_type=task_type, path=path, line=node.lineno)
                for task_type in _task_type_names(keyword.value)
            )
    unique = {
        (producer.task_type, producer.path, producer.line): producer
        for producer in producers
    }
    return sorted(
        unique.values(),
        key=lambda producer: (producer.task_type, producer.path, producer.line),
    )


def _external_facts(tree: ast.Module) -> list[ExternalCall]:
    calls = []
    service_keys = sorted(EXTERNAL_SERVICES, key=lambda key: (-len(key), key))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            service_key = next(
                (
                    key
                    for key in service_keys
                    if module == key or module.startswith(f"{key}.")
                ),
                None,
            )
            if service_key is not None:
                calls.append(
                    ExternalCall(
                        service=EXTERNAL_SERVICES[service_key],
                        module=module,
                        line=node.lineno,
                    )
                )
    unique = {(call.service, call.module, call.line): call for call in calls}
    return sorted(
        unique.values(),
        key=lambda call: (call.service, call.module, call.line),
    )


def _file_facts(
    path: str,
    source: str,
    prefixes: dict[str, frozenset[str]],
    registrations: dict[str, frozenset[tuple[str, str]]],
    webhook_blueprint_paths: frozenset[str],
    queue_deploy_targets: Mapping[str, str],
) -> FileFacts:
    endpoints: list[tuple[str, Endpoint]] = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return FileFacts(
                path=path,
                api_group=None,
                endpoints=[],
                workers=[],
                externals=[],
            )
        endpoints.extend(_flask_endpoints(tree, path, prefixes, registrations))
        endpoints.extend(_blueprint_endpoints(tree, path, webhook_blueprint_paths))
        endpoints.extend(_fastapi_endpoints(tree, path))
        workers = _worker_facts(tree)
        externals = _external_facts(tree)
        task_types = _task_type_facts(tree, path, queue_deploy_targets)
        task_consumers = _task_consumer_facts(tree, path)
        task_producers = _task_producer_facts(tree, path)
        workflow_starts = sorted(
            {
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"start_workflow", "execute_workflow"}
            }
        )
    else:
        return FileFacts(
            path=path,
            api_group=None,
            endpoints=[],
            workers=[],
            externals=[],
        )
    groups = sorted({group for group, _ in endpoints})
    return FileFacts(
        path=path,
        api_group=groups[0] if groups else None,
        endpoints=sorted(
            {
                endpoint.model_dump_json(): endpoint for _, endpoint in endpoints
            }.values(),
            key=lambda endpoint: (
                endpoint.path,
                endpoint.method,
                endpoint.handler,
                endpoint.line,
            ),
        ),
        workers=sorted(
            {worker.model_dump_json(): worker for worker in workers}.values(),
            key=lambda worker: (worker.kind, worker.name, worker.line),
        ),
        externals=externals,
        task_types=task_types,
        task_consumers=task_consumers,
        task_producers=task_producers,
        workflow_starts=workflow_starts,
    )


@dataclass(frozen=True)
class _FileFactsInputs:
    sources: dict[str, str]
    prefixes: dict[str, frozenset[str]]
    registrations: dict[str, frozenset[tuple[str, str]]]
    webhook_blueprint_paths: frozenset[str]
    queue_deploy_targets: dict[str, str]


def _file_facts_dump(inputs: _FileFactsInputs, path: str) -> dict:
    return _file_facts(
        path,
        inputs.sources.get(path, ""),
        inputs.prefixes,
        inputs.registrations,
        inputs.webhook_blueprint_paths,
        inputs.queue_deploy_targets,
    ).model_dump()


def _extract_file_facts(
    repo: Path,
    commit: str,
    requested: list[str],
) -> dict[str, FileFacts]:
    tree = dict(_git_tree(repo, commit))
    missing = sorted(path for path in requested if path not in tree)
    if missing:
        raise ValueError(f"paths are missing at {commit}: {', '.join(missing)}")
    config_paths = {
        "backend/src/rox_core/api/register_namespaces.py",
        "backend/src/rox_core/__init__.py",
        "backend/src/util/listener_utils.py",
    }
    base_paths = set(requested) | (config_paths & tree.keys())
    base_sources = _git_blobs(
        repo,
        [(path, tree[path]) for path in sorted(base_paths) if path.endswith(".py")],
    )
    blueprint_path = "backend/src/rox_core/__init__.py"
    blueprint_tree = (
        ast.parse(base_sources[blueprint_path], filename=blueprint_path)
        if blueprint_path in base_sources
        else ast.Module(body=[], type_ignores=[])
    )
    webhook_blueprint_paths = _webhook_blueprint_paths(blueprint_tree)
    webhook_paths = sorted(
        path
        for path in webhook_blueprint_paths & tree.keys()
        if path.endswith(".py") and path not in base_sources
    )
    sources = {
        **base_sources,
        **_git_blobs(repo, [(path, tree[path]) for path in webhook_paths]),
    }
    prefixes = _cached_namespace_prefixes(repo.resolve(), commit)
    namespace_path = "backend/src/rox_core/api/register_namespaces.py"
    queue_config_path = "backend/src/util/listener_utils.py"
    namespace_tree = (
        ast.parse(sources[namespace_path], filename=namespace_path)
        if namespace_path in sources
        else ast.Module(body=[], type_ignores=[])
    )
    queue_config_tree = (
        ast.parse(sources[queue_config_path], filename=queue_config_path)
        if queue_config_path in sources
        else ast.Module(body=[], type_ignores=[])
    )
    registrations: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for name, targets in _namespace_registration_targets(namespace_tree).items():
        registrations[name].update(targets)
    for path in webhook_paths:
        try:
            webhook_tree = ast.parse(sources[path], filename=path)
        except SyntaxError:
            continue
        for namespace, registered_paths in _registered_paths(webhook_tree).items():
            registrations[namespace].update(
                (prefix, "WEBHOOK") for prefix in registered_paths
            )
    inputs = _FileFactsInputs(
        sources=sources,
        prefixes=prefixes,
        registrations={
            name: frozenset(targets) for name, targets in registrations.items()
        },
        webhook_blueprint_paths=webhook_blueprint_paths,
        queue_deploy_targets=_queue_deploy_targets(queue_config_tree),
    )
    dumps = parallel_map(
        _file_facts_dump,
        inputs,
        requested,
        min_items=PARALLEL_MIN_FILES,
        chunksize=PARALLEL_FILE_CHUNKSIZE,
        desc="Extracting file facts",
        unit="file",
    )
    return {
        path: FileFacts.model_validate(dump)
        for path, dump in zip(requested, dumps, strict=True)
    }


def extract_file_facts(
    repo: Path,
    commit: str,
    paths: Collection[str],
) -> dict[str, FileFacts]:
    requested = sorted(set(paths))
    paths_digest = hashlib.sha256(
        json.dumps(requested, separators=(",", ":")).encode()
    ).hexdigest()
    return get_or_compute(
        commit,
        f"file-facts-{paths_digest}",
        lambda: _extract_file_facts(repo, commit, requested),
    )
