"""Import OpenAPI, Swagger, Postman, Insomnia, HAR, and raw HTTP into APISpec."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from app.domain.http import HttpExchange, HttpHeader
from app.security_testing.api_spec import (
    APISpec,
    Endpoint,
    Parameter,
    RequestSchema,
    ResponseSchema,
)
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState


def import_api(
    source: str | Path | dict[str, object], *, source_format: str | None = None
) -> APISpec:
    if isinstance(source, dict):
        data = source
        text = json.dumps(source)
        path_hint = ""
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        path_hint = path.name.lower()
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"API import requires JSON for this phase: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("API import JSON must be an object")
        data = parsed

    fmt = (source_format or _detect_format(data, path_hint, text)).lower()
    if fmt in {"openapi", "swagger"}:
        return _from_openapi(data)
    if fmt == "postman":
        return _from_postman(data)
    if fmt == "insomnia":
        return _from_insomnia(data)
    if fmt == "har":
        return _from_har(data)
    raise ValueError(f"Unsupported API source format {fmt!r}")


def import_har_exchanges(source: str | Path | dict[str, object]) -> tuple[HttpExchange, ...]:
    loaded: object
    if isinstance(source, dict):
        loaded = source
    else:
        loaded = json.loads(Path(source).read_text(encoding="utf-8"))
    data = loaded if isinstance(loaded, dict) else {}
    log = data.get("log")
    log_dict = log if isinstance(log, dict) else {}
    raw_entries = log_dict.get("entries", [])
    entries = raw_entries if isinstance(raw_entries, list) else []
    exchanges: list[HttpExchange] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        request = entry.get("request", {})
        response = entry.get("response", {})
        if not isinstance(request, dict):
            continue
        url = str(request.get("url", ""))
        method = str(request.get("method", "GET"))
        req_headers = _headers_from_list(request.get("headers", []))
        resp = response if isinstance(response, dict) else {}
        resp_headers = _headers_from_list(resp.get("headers", []))
        content = resp.get("content", {}) if isinstance(resp.get("content"), dict) else {}
        body = content.get("text") if isinstance(content, dict) else None
        post = request.get("postData") if isinstance(request.get("postData"), dict) else {}
        req_body = post.get("text") if isinstance(post, dict) else None
        exchanges.append(
            HttpExchange(
                method=method,
                url=url,
                request_headers=req_headers,
                request_body=str(req_body) if req_body else None,
                response_status=int(resp.get("status") or 0) or None,
                response_headers=resp_headers,
                response_body=str(body) if body else None,
                source_tool="har",
            )
        )
    return tuple(exchanges)


def import_raw_http(text: str) -> HttpExchange:
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines:
        raise ValueError("Empty raw HTTP")
    parts = lines[0].split()
    if len(parts) < 2:
        raise ValueError("Malformed HTTP request line")
    method, target = parts[0], parts[1]
    headers: list[HttpHeader] = []
    idx = 1
    while idx < len(lines) and lines[idx]:
        name, _, value = lines[idx].partition(":")
        headers.append(HttpHeader(name=name.strip(), value=value.strip()))
        idx += 1
    body = "\n".join(lines[idx + 1 :]) if idx + 1 < len(lines) else None
    host = next((h.value for h in headers if h.name.lower() == "host"), "localhost")
    url = target if target.startswith("http") else f"http://{host}{target}"
    return HttpExchange(method=method, url=url, request_headers=tuple(headers), request_body=body)


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _detect_format(data: dict[str, object], path_hint: str, text: str) -> str:
    if "openapi" in data or "swagger" in data:
        return "openapi"
    if data.get("info") and isinstance(data.get("item"), list):
        return "postman"
    if data.get("_type") == "export" or (
        isinstance(data.get("resources"), list) and "insomnia" in text[:200].lower()
    ):
        return "insomnia"
    if "log" in data and isinstance(data.get("log"), dict):
        return "har"
    if path_hint.endswith(".har"):
        return "har"
    return "openapi"


def _from_openapi(data: dict[str, object]) -> APISpec:
    info = _as_dict(data.get("info"))
    title = str(info.get("title") or "imported")
    version = str(info.get("version") or "0.0.0")
    servers = _as_list(data.get("servers"))
    base = ""
    if servers and isinstance(servers[0], dict):
        base = str(servers[0].get("url") or "")
    host = data.get("host")
    if not base and isinstance(host, str):
        scheme = "https"
        schemes = data.get("schemes")
        if isinstance(schemes, list) and schemes:
            scheme = str(schemes[0])
        base_path = str(data.get("basePath") or "")
        base = f"{scheme}://{host}{base_path}"
    paths = _as_dict(data.get("paths"))
    endpoints: list[Endpoint] = []
    for path, ops in paths.items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.startswith("x-") or method == "parameters" or not isinstance(op, dict):
                continue
            params = _parameters(_as_list(op.get("parameters")))
            responses = _responses(_as_dict(op.get("responses")))
            tags = tuple(str(t) for t in _as_list(op.get("tags")))
            endpoints.append(
                Endpoint(
                    method=method.upper(),
                    path=str(path),
                    operation_id=str(op["operationId"]) if op.get("operationId") else None,
                    summary=str(op.get("summary") or ""),
                    parameters=params,
                    request=RequestSchema(parameters=params),
                    responses=responses,
                    tags=tags,
                )
            )
    return APISpec(
        title=title,
        version=version,
        base_url=base,
        endpoints=tuple(endpoints),
        source_format="openapi",
    )


def _from_postman(data: dict[str, object]) -> APISpec:
    info = _as_dict(data.get("info"))
    title = str(info.get("name") or "postman")
    endpoints = tuple(_walk_postman(_as_list(data.get("item"))))
    return APISpec(title=title, endpoints=endpoints, source_format="postman")


def _walk_postman(items: list[object]) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        nested = item.get("item")
        if isinstance(nested, list):
            endpoints.extend(_walk_postman(nested))
            continue
        request = item.get("request")
        if not isinstance(request, dict):
            continue
        method = str(request.get("method") or "GET")
        url_obj = request.get("url")
        path = "/"
        if isinstance(url_obj, str):
            path = urlparse(url_obj).path or url_obj
        elif isinstance(url_obj, dict):
            raw = url_obj.get("raw")
            if isinstance(raw, str) and "://" in raw:
                path = urlparse(raw).path or "/"
            else:
                parts = url_obj.get("path")
                if isinstance(parts, list):
                    path = "/" + "/".join(str(p) for p in parts)
        endpoints.append(
            Endpoint(method=method.upper(), path=path, summary=str(item.get("name") or ""))
        )
    return endpoints


def _from_insomnia(data: dict[str, object]) -> APISpec:
    resources = _as_list(data.get("resources"))
    endpoints: list[Endpoint] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if resource.get("_type") != "request":
            continue
        method = str(resource.get("method") or "GET")
        url = str(resource.get("url") or "/")
        path = urlparse(url).path if "://" in url else url
        endpoints.append(
            Endpoint(
                method=method.upper(), path=path or "/", summary=str(resource.get("name") or "")
            )
        )
    title = str(data.get("name") or "insomnia")
    return APISpec(title=title, endpoints=tuple(endpoints), source_format="insomnia")


def _from_har(data: dict[str, object]) -> APISpec:
    exchanges = import_har_exchanges(data)
    endpoints = tuple(
        Endpoint(method=ex.method, path=ex.path or "/", summary=ex.url) for ex in exchanges
    )
    return APISpec(title="har-import", endpoints=endpoints, source_format="har")


def _parameters(raw: list[object]) -> tuple[Parameter, ...]:
    params: list[Parameter] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        schema = _as_dict(item.get("schema"))
        params.append(
            Parameter(
                name=str(item.get("name") or ""),
                location=str(item.get("in") or "query"),
                required=bool(item.get("required")),
                schema_type=str(schema.get("type") or item.get("type") or "string"),
            )
        )
    return tuple(params)


def _responses(raw: dict[str, object]) -> tuple[ResponseSchema, ...]:
    out: list[ResponseSchema] = []
    for code, body in raw.items():
        try:
            status = int(code)
        except ValueError:
            status = 0
        desc = ""
        if isinstance(body, dict):
            desc = str(body.get("description") or "")
        out.append(ResponseSchema(status=status, description=desc))
    return tuple(out)


def _headers_from_list(raw: object) -> tuple[HttpHeader, ...]:
    if not isinstance(raw, list):
        return ()
    headers: list[HttpHeader] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            headers.append(HttpHeader(name=str(item["name"]), value=str(item.get("value") or "")))
    return tuple(headers)


def malformed_import(detail: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool="api_import", state=ToolExecutionState.MALFORMED_INPUT, detail=detail
    )
