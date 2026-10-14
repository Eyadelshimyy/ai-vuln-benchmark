"""
A pipeline as data: ``pipelines/<name>.yaml`` (also ``.yml``).

A pipeline file only *names* things: the phase-① transform to run, the models
it feeds, the reports to rebuild, and an optional schedule. It compiles to the
runner :func:`tracebi.pipeline.model_pipeline.model_pipeline` builds, so
``tracebi run-pipeline`` and the Refresh page behave exactly as they do for
the Python form: run the transform fresh, then rebuild the reports, recording
each run. The transform stays Python; nothing here can express code.

The schema is closed, the file loads with ``yaml.safe_load`` semantics, and a
repeated key is an error (the model loader's rules).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from tracebi.model.model_spec import (
    ModelSpecError, _NAME, _unknown, _unique_key_loader,
)

SUFFIXES = (".yaml", ".yml")
_KEYS = {"name", "description", "transform", "models", "reports", "schedule"}
_TRANSFORM = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_CRON_FIELD = re.compile(r"[0-9A-Za-z*/,\-?]+")


def read_pipeline_doc(path: "str | os.PathLike[str]") -> Any:
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML pipelines need PyYAML: pip install pyyaml") from exc
    try:
        return yaml.load(path.read_text(encoding="utf-8"),  # noqa: S506 — a SafeLoader subclass
                         Loader=_unique_key_loader())
    except yaml.YAMLError as exc:
        raise ModelSpecError(path.name, [f"not valid YAML ({' '.join(str(exc).split())})"]) from exc


def _strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(v, str) and v for v in value)


def validate_pipeline_spec(doc: Any) -> list[str]:
    """Structural errors in a pipeline document, each led by its key."""
    if not isinstance(doc, dict):
        return ["the file must hold one object (a mapping at the top level)"]
    errors = _unknown(doc, _KEYS, "pipeline")
    if not isinstance(doc.get("name"), str) or not _NAME.fullmatch(doc.get("name") or ""):
        errors.append("name: required, letters, digits and underscores")
    transform = doc.get("transform")
    if not isinstance(transform, str) or not _TRANSFORM.fullmatch(transform) \
            or ".." in transform:
        errors.append("transform: required, the name of a transforms/<x>.py "
                      "or .ipynb (a bare name, no folder)")
    elif "." in transform and not transform.endswith((".py", ".ipynb")):
        errors.append(f"transform: '{transform}' is not a .py or .ipynb")
    if not _strings(doc.get("models")):
        errors.append("models: required, a list of model names")
    reports = doc.get("reports", "all")
    if reports != "all" and not (
            _strings(reports) and not any(".." in r or r.startswith("/") for r in reports)):
        errors.append("reports: must be \"all\" (every report of the models) or "
                      "a list of report names")
    schedule = doc.get("schedule")
    if schedule is not None:
        fields = schedule.split() if isinstance(schedule, str) else []
        if len(fields) != 5 or not all(_CRON_FIELD.fullmatch(f) for f in fields):
            errors.append("schedule: must be a 5-field cron expression "
                          "(minute hour day month weekday) or null")
    description = doc.get("description")
    if description is not None and not isinstance(description, str):
        errors.append("description: must be a string")
    return errors


def load_pipeline_spec(path: "str | os.PathLike[str]"):
    """Compile ``pipelines/<name>.yaml`` into a runner. The name must equal
    the file stem. Runs nothing and reads no warehouse."""
    from tracebi.pipeline.model_pipeline import model_pipeline

    path = Path(path)
    doc = read_pipeline_doc(path)
    errors = validate_pipeline_spec(doc)
    if not errors and doc["name"] != path.stem:
        errors.append(f"name: must equal the file name '{path.stem}', got {doc['name']!r}")
    if errors:
        raise ModelSpecError(path.name, errors)
    transform = doc["transform"]
    for ext in (".py", ".ipynb"):
        if transform.endswith(ext):
            transform = transform[: -len(ext)]
    reports = doc.get("reports", "all")
    runner = model_pipeline(
        doc["models"][0], transform, models=doc["models"],
        reports=None if reports == "all" else reports,
        schedule=doc.get("schedule"), name=doc["name"])
    runner.description = doc.get("description")
    return runner


def validate_pipeline_file(path: "str | os.PathLike[str]") -> list[str]:
    """The structural problems of a pipeline file; empty means it is well formed."""
    try:
        path = Path(path)
        doc = read_pipeline_doc(path)
        errors = validate_pipeline_spec(doc)
        if not errors and doc["name"] != path.stem:
            errors.append(f"name: must equal the file name '{path.stem}', got {doc['name']!r}")
        return errors
    except ModelSpecError as exc:
        return exc.errors
    except Exception as exc:  # noqa: BLE001 — the answer is the message
        return [str(exc)]


def pipeline_to_yaml(runner, stem: str) -> str:
    """The YAML for a Python pipeline's loaded runner, or ``ValueError`` with
    the reason it cannot be written declaratively."""
    declared = getattr(runner, "declared", None)
    if not declared:
        raise ValueError(
            "this runner was not built by model_pipeline(); only a "
            "transform -> build pipeline can be a pipelines/<name>.yaml. "
            "Medallion layers and custom steps stay Python")
    extra = [layer["name"] for layer in runner.layers()
             if layer["name"] not in ("transform", "build")]
    if extra:
        raise ValueError(f"the runner also registers {', '.join(extra)}; a YAML "
                         f"pipeline is exactly transform then build")
    if declared["custom_db_url"]:
        raise ValueError("the runner keeps its run history at a custom db_url, "
                         "which a YAML pipeline cannot name (it uses "
                         "data/<name>_runs.db)")
    if declared["reports_dir"] != "reports":
        raise ValueError(f"the runner builds reports from {declared['reports_dir']!r}; "
                         f"a YAML pipeline uses reports/")
    if declared["name"] != stem:
        # model_pipeline(model, ...) names its history file after the model, so
        # a pipeline file named differently would move that history.
        raise ValueError(f"the runner's run history is data/{declared['name']}_runs.db "
                         f"but the file is {stem!r}; rename one so they match")
    lines = [f"# {stem}: generated by `tracebi migrate pipeline` from pipelines/{stem}.py.",
             f"# While pipelines/{stem}.py exists it wins over this file: delete one of them.",
             f"name: {stem}"]
    description = (getattr(runner, "description", None) or "").strip()
    if description:
        lines.append(f"description: {json.dumps(description)}")
    lines += [
        "",
        "# transform: the phase-1 script to run first (transforms/<name>.py or .ipynb)",
        f"transform: {declared['transform']}",
        "",
        "# models: what the transform feeds",
        "models: [" + ", ".join(declared["models"]) + "]",
        "",
        "# reports: \"all\" rebuilds every report of those models, or list names",
    ]
    if declared["reports"] is None:
        lines.append("reports: all")
    else:
        lines.append("reports:")
        lines += [f"  - {r}" for r in declared["reports"]]
    lines += ["", "# schedule: a 5-field cron expression, or null to run on demand",
              f"schedule: {json.dumps(declared['schedule'])}" if declared["schedule"] else "schedule: null"]
    text = "\n".join(lines) + "\n"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / f"{stem}.yaml"
        probe.write_text(text, encoding="utf-8")
        errors = validate_pipeline_file(probe)
    if errors:
        raise ValueError("the generated YAML is not valid: " + "; ".join(errors))
    return text
