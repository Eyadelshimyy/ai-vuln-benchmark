"""Named starters for ``tracebi init --template``.

The copy logic lives here so ``cli.py`` only branches on the flag. Each
starter is a directory of files under ``tracebi/_scaffold/templates/``
(sample data, transform, model, report packages). Shared project files
— gitignore, env example, ``AGENTS.md``, MCP config — are the same ones
default ``init`` writes.
"""

from __future__ import annotations

import json
import sys
from importlib.resources import files
from pathlib import Path

# Public name → bundle directory under ``_scaffold/templates/``.
_BUNDLES: dict[str, str] = {
    "saas-metrics": "saas_metrics",
    "sales-pipeline": "sales_pipeline",
}

# Accepted spellings → public name.
_ALIASES: dict[str, str] = {
    "saas-metrics": "saas-metrics",
    "saas_metrics": "saas-metrics",
    "sales-pipeline": "sales-pipeline",
    "sales_pipeline": "sales-pipeline",
}

_CATALOG: list[dict] = [
    {
        "name": "saas-metrics",
        "aliases": ["saas_metrics"],
        "summary": (
            "Account-month subscriptions: ending MRR, logo churn "
            "(a ratio of totals), and a signup-cohort cut."
        ),
        "model": "saas_model",
        "reports": [
            "saas_model/mrr_dashboard",
            "saas_model/cohort_brief",
        ],
        "sample": "inputs/subscriptions.csv",
        "transform": "transforms/saas_transform.py",
        "next": (
            "tracebi connect <name> --kind postgres|snowflake|bigquery|duckdb, "
            "then point models/saas_model.yaml's `connection:` at it."
        ),
    },
    {
        "name": "sales-pipeline",
        "aliases": ["sales_pipeline"],
        "summary": (
            "Opportunities by stage, rep, and region: open pipeline "
            "value, and win rate as won deals over closed deals (a ratio "
            "of totals)."
        ),
        "model": "sales_pipeline_model",
        "reports": [
            "sales_pipeline_model/pipeline_dashboard",
            "sales_pipeline_model/rep_scorecard",
        ],
        "sample": "inputs/opportunities.csv",
        "transform": "transforms/sales_pipeline_transform.py",
        "next": (
            "tracebi connect <name> --kind postgres|snowflake|bigquery|duckdb, "
            "then point models/sales_pipeline_model.yaml's `connection:` at it."
        ),
    },
]


def known_template_names() -> list[str]:
    """Public template names, sorted. This is the refusal list."""
    return sorted(_BUNDLES)


def resolve_template(name: str) -> str | None:
    """Map a flag value to its public name, or None when it is unknown."""
    return _ALIASES.get(name.strip())


def template_catalog() -> list[dict]:
    """The ``templates.known`` list in ``tracebi context``."""
    return [dict(entry) for entry in _CATALOG]


def _scaffold_text(name: str) -> str:
    return files("tracebi").joinpath("_scaffold", name).read_text(encoding="utf-8")


def _mcp_json() -> str:
    return json.dumps(
        {"mcpServers": {"tracebi": {"command": "tracebi", "args": ["mcp"]}}},
        indent=2,
    ) + "\n"


def _bundle_files(bundle_id: str) -> dict[Path, str]:
    root = files("tracebi").joinpath("_scaffold", "templates", bundle_id)
    out: dict[Path, str] = {}

    def walk(node, prefix: Path) -> None:
        for child in sorted(node.iterdir(), key=lambda p: p.name):
            if child.name in {"__pycache__", ".DS_Store"}:
                continue
            rel = prefix / child.name
            if child.is_dir():
                walk(child, rel)
            elif child.is_file():
                out[rel] = child.read_text(encoding="utf-8")

    walk(root, Path())
    return out


def _pipeline(model: str, transform: str) -> str:
    """``pipelines/<model>.yaml``: the transform, then the model's reports — what
    ``tracebi run-pipeline`` and the app's Refresh page run."""
    return f"""\
# Pipeline for {model}: rebuild the warehouse, then its reports.
#
#     transform  runs {transform}, which sinks the star schema
#     build      builds every report in reports/{model}/, each with a receipt
#
#     tracebi run-pipeline {model}
name: {model}
description: Rebuild the warehouse, then its reports.
transform: {Path(transform).stem}
models: [{model}]
reports: all
schedule: null
"""


def _readme(project: str, template: str) -> str:
    entry = next(item for item in _CATALOG if item["name"] == template)
    verifies = "\n".join(
        f"tracebi verify output/{name}.html.manifest.json --strict --contracts"
        for name in entry["reports"]
    )
    report_lines = "\n".join(f"├── reports/{name}/" for name in entry["reports"])
    return f"""\
# {project}

A TraceBi project. Scaffolded by `tracebi init --template {template}`.

{entry["summary"]}
The file `{entry["sample"]}` is a small synthetic sample so the loop runs
before you have a warehouse.

## Next step — your own data

1. `{entry["next"].split(", then ")[0]}`
   tests the warehouse, writes the secret to `.env`, and writes
   `connections/<name>.yaml` (the secret appears there only as `${{ENV_VAR}}`).
2. Re-point `models/{entry["model"]}.yaml` at those tables (`connection:`
   and the `source:` names). Or draft a fresh model with
   `tracebi new-model "<Name>" --from <name> --tables ...` and edit every
   line marked `# DRAFT: review` before a report depends on it.

## Run the sample loop

```bash
tracebi run-pipeline {entry["model"]}   # the transform, then the model's reports
{verifies}
tracebi serve
```

`verify` re-runs the recorded queries. `--contracts` re-runs the sink
contract. The claim on the transform is "the sink satisfied its contract".

## Layout

```
{project}/
├── {entry["sample"]}
├── {entry["transform"]}
├── connections/warehouse.yaml
├── models/{entry["model"]}.yaml
├── pipelines/{entry["model"]}.yaml
{report_lines}
├── data/             warehouse (gitignored)
└── output/           rendered HTML + manifest receipts
```
"""


def init_template_project(project: Path, template: str, *, force: bool) -> int:
    """Scaffold *project* from *template*. Same empty-dir / ``--force`` rules as init."""
    canonical = resolve_template(template)
    if canonical is None:
        known = ", ".join(known_template_names())
        print(
            f"unknown template {template!r}. Known templates: {known}.",
            file=sys.stderr,
        )
        return 1

    target = Path(project).resolve()
    if target.exists() and any(target.iterdir()):
        if not force:
            print(
                f"refusing to init into non-empty {target}; "
                f"pass --force to override",
                file=sys.stderr,
            )
            return 1

    for d in ("inputs", "transforms", "connections", "models", "pipelines",
              "reports", "data", "output"):
        (target / d).mkdir(parents=True, exist_ok=True)

    entry = next(item for item in _CATALOG if item["name"] == canonical)
    to_write: dict[Path, str] = {
        target / ".gitignore": _scaffold_text("init_gitignore.txt"),
        target / ".env.example": _scaffold_text("init_env_example.txt"),
        target / "README.md": _readme(target.name, canonical),
        target / "AGENTS.md": _scaffold_text("init_agents.md"),
        target / ".mcp.json": _mcp_json(),
        target / ".cursor" / "mcp.json": _mcp_json(),
        target / "connections" / "warehouse.yaml": _scaffold_text("init_connection_warehouse.yaml.txt"),
        target / "pipelines" / f"{entry['model']}.yaml": _pipeline(entry["model"], entry["transform"]),
    }
    for rel, content in _bundle_files(_BUNDLES[canonical]).items():
        to_write[target / rel] = content

    for path, content in to_write.items():
        if path.exists() and not force:
            print(f"skipping existing {path}", file=sys.stderr)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Initialised TraceBi project at {target} (template {canonical})")
    print(f"  cd {target.name}")
    print(f"  tracebi run-pipeline {entry['model']}   # the transform, then the model's reports")
    first = entry["reports"][0]
    print(
        f"  tracebi verify output/{first}.html.manifest.json "
        f"--strict --contracts"
    )
    print("  tracebi serve")
    print(f"  Next: {entry['next']}")
    return 0
