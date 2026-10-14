from fastapi import APIRouter, HTTPException
from tracebi.web.api.registry import registry
from tracebi.web.api.source_view import source_payload

router = APIRouter(prefix="/connectors", tags=["connectors"])


@router.get("")
def list_connectors():
    """List all registered connectors."""
    return registry.list_connectors()


@router.get("/{name}")
def get_connector(name: str):
    """Get detail for a single connector."""
    c = registry.get_connector(name)
    if not c:
        raise HTTPException(status_code=404, detail=f"Connector '{name}' not found")
    return c.describe()


@router.get("/{name}/source")
def connector_source(name: str):
    """Where the connector is declared: the model file(s) that read from it."""
    connector = registry.get_connector(name)
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector '{name}' not found")
    from tracebi.model_registry import model_path

    used_by = next((c.get("used_by", []) for c in registry.list_connectors() if c["name"] == name), [])
    # A declared connection file holds ${ENV_VAR} references only (a literal
    # secret does not validate), so showing it shows no credential.
    declared = getattr(connector, "declared_in", None)
    return source_payload(
        [("connection", declared)] + [(f"model {m}", model_path(m)) for m in used_by],
        missing_hint="No model file declares this connector: it is registered in Python code (an app module).",
        hint="A connector is declared in connections/<name>.yaml, or in the model file that reads from it.",
    )


@router.get("/{name}/warehouse")
def connector_warehouse(name: str):
    """What a file-backed warehouse holds: tables, row counts, column profiles
    and each table's sink-contract status. Read-only; the file is opened and
    closed within the request. Other connectors answer ``supported: false``."""
    c = registry.get_connector(name)
    if not c:
        raise HTTPException(status_code=404, detail=f"Connector '{name}' not found")
    d = c.describe()
    database = d.get("database")
    if d.get("type") != "DuckDBConnector" or not database or database == ":memory:":
        return {"supported": False, "tables": []}
    from tracebi.workbench import warehouse_view

    return {"supported": True, **warehouse_view(str(database))}
