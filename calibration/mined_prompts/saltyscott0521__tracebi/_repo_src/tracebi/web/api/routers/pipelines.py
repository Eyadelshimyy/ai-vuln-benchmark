from typing import Optional

from fastapi import APIRouter, HTTPException

from tracebi.web.api import pipeline_runs
from tracebi.web.api.registry import registry
from tracebi.web.api.source_view import source_payload

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("")
def list_pipelines():
    """List all registered pipeline runners and their layer status."""
    result = []
    for pipeline_name in registry.list_pipeline_names():
        runner = registry.get_pipeline(pipeline_name)
        layers = []
        for layer in runner.layers():
            try:
                last = runner.last_run(layer["name"])
            except Exception as exc:
                last = None
                layer["last_status"] = f"error: {exc}"
            if last is not None:
                layer["last_status"] = last["status"]
                layer["last_run"] = str(last["completed_at"] or "")
                layer["last_rows_out"] = (
                    int(last["rows_out"]) if last["rows_out"] is not None else None
                )
                # None for runs recorded before attribution existed, and for
                # anything executed without an actor in scope.
                layer["last_actor"] = last.get("actor")
                layer["last_actor_role"] = last.get("actor_role")
            else:
                layer.setdefault("last_status", None)
                layer["last_run"] = None
                layer["last_rows_out"] = None
                layer["last_actor"] = None
                layer["last_actor_role"] = None
            layers.append(layer)
        # model_pipeline stamps .model; a hand-built runner may leave it unset.
        # A pipeline can touch more than one model (it lands data from one and
        # builds another), so a hand-built runner may also set .models.
        model = getattr(runner, "model", None)
        model = model if isinstance(model, str) and model else None
        named = getattr(runner, "models", None)
        models = [m for m in (named or []) if isinstance(m, str) and m]
        # What a pipelines/<name>.yaml says about itself: the transform it runs
        # and its one-line description (the model's Overview shows both).
        transform = getattr(runner, "transform", None)
        description = getattr(runner, "description", None)
        result.append({
            "pipeline": pipeline_name,
            "model": model,
            "models": models or ([model] if model else []),
            "transform": transform if isinstance(transform, str) and transform else None,
            "description": description.strip() if isinstance(description, str) and description.strip() else None,
            "layers": layers,
        })
    return result


def _pipeline_order(runner, refresh: bool) -> list[str]:
    """The layers a whole-pipeline run executes, in order.

    With ``refresh``, each leaf is run with its full upstream chain, so
    dependencies fire in the right order with no duplicates; without it, every
    registered layer fires independently regardless of dependencies.
    """
    layers = runner.layers()
    names = [layer["name"] for layer in layers]
    if not refresh:
        return names
    # Leaves = layers nothing else depends on. Running each leaf's full chain
    # covers every layer in dependency order.
    depends = {layer["depends_on"] for layer in layers if layer["depends_on"]}
    leaves = [n for n in names if n not in depends] or names
    order: list[str] = []
    for leaf in leaves:
        for name in runner.execution_order(leaf):
            if name not in order:
                order.append(name)
    return order


@router.post("/{pipeline_name}/run")
def run_pipeline(pipeline_name: str, refresh: bool = True):
    """
    Run every layer in a pipeline.

    With ``refresh=true`` (default), each leaf is run with its full
    upstream chain, so dependencies fire in the right order with no
    duplicates. Set ``refresh=false`` to fire every registered layer
    independently regardless of dependencies.
    """
    runner = registry.get_pipeline(pipeline_name)
    if not runner:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

    order = _pipeline_order(runner, refresh)
    if not order:
        return {"pipeline": pipeline_name, "status": "empty", "ran": []}

    ran: list[str] = []
    try:
        for name in order:
            runner.execute_layer(name)
            ran.append(name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"pipeline": pipeline_name, "status": "ok", "ran": ran}


@router.post("/{pipeline_name}/layers/{layer_name}/run")
def run_layer(pipeline_name: str, layer_name: str, refresh: bool = False):
    """
    Trigger a pipeline layer on demand.

    Set refresh=true to walk the full depends_on chain upstream first.
    """
    runner = registry.get_pipeline(pipeline_name)
    if not runner:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    if not runner.has_layer(layer_name):
        raise HTTPException(
            status_code=404,
            detail=f"Layer '{layer_name}' not found in pipeline '{pipeline_name}'",
        )
    try:
        runner.run(layer_name, refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"pipeline": pipeline_name, "layer": layer_name, "status": "triggered"}


@router.get("/{pipeline_name}/layers/{layer_name}/history")
def layer_history(pipeline_name: str, layer_name: str, limit: int = 20):
    """Return the run history for a specific layer."""
    runner = registry.get_pipeline(pipeline_name)
    if not runner:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    if not runner.has_layer(layer_name):
        raise HTTPException(
            status_code=404,
            detail=f"Layer '{layer_name}' not found in pipeline '{pipeline_name}'",
        )
    try:
        runs = runner.run_history(layer_name, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "pipeline": pipeline_name,
        "layer": layer_name,
        "runs": runs,
    }


@router.post("/{pipeline_name}/runs", status_code=202)
def start_pipeline_run(pipeline_name: str, layer: Optional[str] = None,
                       refresh: Optional[bool] = None):
    """
    Start a run in the background and return at once; its log is read from
    ``GET /pipelines/{name}/runs/{run_id}/log`` as it grows.

    Without ``layer`` the whole pipeline runs (upstream first, as ``/run`` does);
    with one, just that layer, plus its upstream chain when ``refresh=true``.
    A pipeline already running returns that run, with ``already_running: true``.
    """
    runner = registry.get_pipeline(pipeline_name)
    if not runner:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    if layer is not None and not runner.has_layer(layer):
        raise HTTPException(
            status_code=404,
            detail=f"Layer '{layer}' not found in pipeline '{pipeline_name}'",
        )
    if refresh is None:
        refresh = layer is None
    if layer is None:
        order = _pipeline_order(runner, refresh)
    else:
        order = runner.execution_order(layer) if refresh else [layer]
    if not order:
        raise HTTPException(status_code=422, detail=f"Pipeline '{pipeline_name}' has no layers to run")
    return pipeline_runs.start(pipeline_name, runner, order)


@router.get("/{pipeline_name}/runs")
def pipeline_run_history(pipeline_name: str, limit: int = 10):
    """Recent background runs of this pipeline, newest first."""
    if not registry.get_pipeline(pipeline_name):
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    return pipeline_runs.history(pipeline_name, limit)


@router.get("/{pipeline_name}/runs/{run_id}")
def pipeline_run_status(pipeline_name: str, run_id: str):
    """Status of one background run."""
    run = pipeline_runs.get(pipeline_name, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found for pipeline '{pipeline_name}'")
    return run


@router.get("/{pipeline_name}/runs/{run_id}/log")
def pipeline_run_log(pipeline_name: str, run_id: str, after: int = 0):
    """What the run printed, from byte ``after``. Ask again from ``next`` until ``done``.

    Needs ``analyst``: a log is whatever the steps printed, not only their status.
    """
    log = pipeline_runs.read_log(pipeline_name, run_id, after)
    if log is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found for pipeline '{pipeline_name}'")
    return log


@router.get("/{pipeline_name}/source")
def pipeline_source(pipeline_name: str):
    """The code a refresh runs: the pipeline file, and the transform its model
    pipeline runs first."""
    runner = registry.get_pipeline(pipeline_name)
    if not runner:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    from tracebi.cli import find_transform
    from tracebi.pipeline_registry import pipeline_path

    transform = getattr(runner, "transform", None)
    found = find_transform(transform) if isinstance(transform, str) else None
    return source_payload(
        [("pipeline", pipeline_path(pipeline_name)), ("transform", str(found) if found else None)],
        missing_hint="This pipeline is registered in Python code (an app module), not from a pipelines/ file.",
        hint="Run all runs the pipeline's steps in order; the transform is the first, and writes the warehouse.",
    )
