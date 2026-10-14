"""
TraceBi CLI.

Scaffolds and drives a project through the three-phase workflow — TRANSFORM
(``transforms/``) → MODEL (``models/``) → REPORT (``reports/``):

    tracebi init my_project                 # scaffold a new project
    tracebi run-transform <name>            # ① run a transform → sink the warehouse
    tracebi connect wh --kind duckdb        # point at a warehouse you already have
    tracebi import dbt <path>               # draft a model from a dbt manifest.json
    tracebi new-model "Sales Model"         # ② scaffold a model over the warehouse
    tracebi report build <name>             # ③ render an artifact package + receipt
    tracebi verify <manifest>               # re-run recorded queries; classify drift
    tracebi serve                           # browse the project
    tracebi mcp                             # agent gateway over MCP
    tracebi --version

Run ``tracebi --help`` for the full command list. The CLI stays thin — it
scaffolds files and drives the library; the behaviour lives in the library.
Its scaffold templates are data files under ``tracebi/_scaffold/`` (loaded by
:func:`_scaffold_text`), not inline string constants.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import runpy
import shutil
import sys
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Optional

from tracebi._version import get_version as _tracebi_version


# Resolve the project folders relative to the user's current working
# directory by default, honouring the same TRACEBI_*_DIR env vars the server
# and gateway read (so the CLI and the server agree on where a project
# lives). Override per-invocation with the --*-dir flags.
def _scaffold_text(name: str) -> str:
    """Read a bundled scaffold template from ``tracebi/_scaffold/``.

    The init scaffold's files live as data files there rather than as
    inline string constants, so they read and diff as the file type they
    are. Anchored on the ``tracebi`` package so it resolves in a wheel.
    """
    return files("tracebi").joinpath("_scaffold", name).read_text(encoding="utf-8")


def _default_models_dir() -> Path:
    return Path(os.environ.get("TRACEBI_MODELS_DIR", "models"))


def _default_connections_dir() -> Path:
    return Path(os.environ.get("TRACEBI_CONNECTIONS_DIR", "connections"))


def _default_pipelines_dir() -> Path:
    return Path(os.environ.get("TRACEBI_PIPELINES_DIR", "pipelines"))


def _default_reports_dir() -> Path:
    return Path(os.environ.get("TRACEBI_REPORTS_DIR", "reports"))


def _default_transforms_dir() -> Path:
    return Path(os.environ.get("TRACEBI_TRANSFORMS_DIR", "transforms"))


def _slugify(title: str) -> str:
    """Convert "Open orders by region" → "open_orders_by_region"."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "report"


# The declarative model forms (see tracebi.model.model_spec.SPEC_SUFFIXES);
# repeated here so the CLI does not import pandas to list a folder.
_MODEL_FILE_SUFFIXES = (".yaml", ".yml", ".json")
_PIPELINE_FILE_SUFFIXES = (".yaml", ".yml")


def _pipeline_yaml_template_text(title: str) -> str:
    today = date.today().isoformat()
    slug = _slugify(title)
    return f"""\
# {title}
#
# A pipeline: run a phase-1 transform, then rebuild the reports. Scaffolded by
# `tracebi new-pipeline` on {today}. It only NAMES things; the transform stays
# Python. Run it with `tracebi run-pipeline {slug}`, or with Refresh data on the model's Overview.
#
# If a pipelines/{slug}.py exists too, the Python file wins (a pipeline is one file).

# The name equals this file's name.
name: {slug}
description: Rebuild the warehouse, then its reports.

# transform: the script to run first, transforms/<name>.py (or .ipynb), run
# top-to-bottom in a fresh namespace. `tracebi new-transform` scaffolds one.
transform: {slug}

# models: the models that transform feeds (models/<name>.yaml or .py).
models: [{slug}]

# reports: "all" rebuilds every report in reports/<model>/ for those models, or
# list names to rebuild only some:
#   reports:
#     - {slug}/overview
reports: all

# schedule: a 5-field cron expression (minute hour day month weekday), or null
# to run only on demand. A scheduled run does the transform, then the reports.
schedule: null
"""


def _model_yaml_template_text(title: str) -> str:
    today = date.today().isoformat()
    slug = _slugify(title)
    return f"""\
# {title}
#
# A model: the star schema over the warehouse your phase-1 transform sank.
# Scaffolded by `tracebi new-model` on {today}. Comments are the point of
# YAML: say why a measure is weighted and which trap it avoids.
#
# Use it from a notebook or script:
#     from tracebi.model_registry import get_model
#     model = get_model("{slug}")
#
# Files in models/ are discovered by the web server and by get_model(). Loading
# reads no rows; a query is what opens the warehouse. If a models/{slug}.py
# exists too, the Python file wins (a model is one file).

# The name equals this file's name.
name: {slug}

# connectors: where the tables live. `database` is a DuckDB file inside the
# project (no absolute path, no ".."). No credentials belong here: for a
# database, run `tracebi connect` and use `connection: <name>` instead.
connectors:
  - name: warehouse
    type: duckdb
    database: data/warehouse.duckdb

# tables: each table the transform sank, by the name it has in the warehouse.
tables:
  - name: my_table
    connector: warehouse
    source: my_table
  # - name: customers
  #   connector: warehouse
  #   source: customers

# relationships: named joins between tables (right_key defaults to left_key,
# how to left). Only needed for tables the star schema below does not join.
# relationships:
#   - name: orders_customers
#     left_table: orders
#     right_table: customers
#     left_key: customer_id

# dimensions: the descriptive tables. `key` is the column facts point at;
# `attributes` are the columns reports may group and filter by.
# dimensions:
#   - name: dim_customer
#     table: customers
#     key: customer_id
#     attributes: [region, segment]

# facts: the event tables. `measures` lists the numeric columns, and
# `foreign_keys` maps each dimension to this table's key column for it.
# facts:
#   - name: fact_orders
#     table: orders
#     measures: [revenue, qty]
#     foreign_keys:
#       dim_customer: customer_id

# time_grains / value_bins: groupable attributes derived from a dimension
# column, so "by month" or "by score band" is declared once, not re-cut in
# every report.
# time_grains:
#   - dimension: dim_customer
#     name: signup_month
#     source: signup_date
#     grain: month
# value_bins:
#   - dimension: dim_customer
#     name: score_band
#     source: credit_score
#     edges: [600, 700, 800]      # N edges make N+1 bands

# measures: the named vocabulary every report asks for. One of column+agg,
# expr+agg, ratio, share, rank, running, period_end, offset, growth, to_date.
# A ratio is a ratio of totals, not a mean of row ratios.
# measures:
#   - name: revenue
#     column: revenue
#     agg: sum
#     description: Total revenue
#     format: currency0
"""


def _model_template_text(title: str) -> str:
    today = date.today().isoformat()
    slug = _slugify(title)
    model_name = title.strip().replace(" ", "")
    return f'''\
"""
{title}
{'=' * len(title)}

DataModel definition. Scaffolded by ``tracebi new-model`` on {today}.

Use in any notebook or script:

    from tracebi.model_registry import get_model
    model = get_model("{slug}")
    ds = model.load("my_table")

Or set as default so notebooks that call get_default_model() pick it up:

    from tracebi.model_registry import get_model, set_default
    set_default("{slug}")
"""

import os

from tracebi import DataModel
from tracebi.connectors.duckdb_connector import DuckDBConnector
# from tracebi import CSVConnector, SQLConnector, MemoryConnector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Connectors ───────────────────────────────────────────────────────────────
# Default: the warehouse your phase-① transforms sink into. Swap for
# SQLConnector("{slug}_db", url=os.environ["{slug}_DB_URL".upper()]) etc. to
# model a database directly.
connector = DuckDBConnector("warehouse",
                            database=os.path.join(ROOT, "data", "warehouse.duckdb"))

# ── Model ─────────────────────────────────────────────────────────────────────
# The variable MUST be named `model` — the registry looks for it at load time.
model = DataModel("{model_name}")
model.add_connector(connector)
model.add_table("my_table", connector="warehouse", source="my_table")
# model.add_relationship(
#     name="orders_customers",
#     left_table="orders", right_table="customers", left_key="customer_id",
# )
# model.add_dimension("dim_customer", table_name="customers", key_col="customer_id",
#                     attributes=["region", "segment"])
# model.add_fact("fact_orders", table_name="orders", measures=["revenue", "qty"],
#                foreign_keys={{"dim_customer": "customer_id"}})

# Construction is declarative and lazy: importing this file must not touch
# the database (discovery imports every model file, and a connect here would
# open a connection — or fail outright — on every scan). A query is what
# connects. Do not call model.connect() at import time.

# Files in models/ are auto-discovered — by the web server, and by
# `get_model("{slug}")` from notebooks and scripts. An explicit
# `register.model(model)` is only needed for files living OUTSIDE models/.
'''


def _pipeline_template_text(title: str) -> str:
    today = date.today().isoformat()
    slug = _slugify(title)
    return f'''\
"""
{title}
{'=' * len(title)}

PipelineRunner definition. Scaffolded by ``tracebi new-pipeline`` on {today}.

Use in any script or notebook:

    from tracebi.pipeline_registry import get_runner
    runner = get_runner("{slug}")
    runner.run("layer_name")
    runner.status()
"""

from tracebi import PipelineRunner
# from tracebi import LandingLayer, ManipulationLayer, FinalLayer, SQLConnector
# from tracebi.model_registry import get_model

_DB_URL = "sqlite:///data/{slug}.db"

# ── Layers ────────────────────────────────────────────────────────────────────
# Uncomment and adapt to wire your medallion layers.
#
# from tracebi import LandingLayer, ManipulationLayer, SQLConnector
# _db = SQLConnector("{slug}_db", url=_DB_URL)
#
# _bronze = LandingLayer(
#     connector=_db, source="orders_raw",
#     sink=_db, sink_table="orders_bronze",
# )
# _silver = (
#     ManipulationLayer(source=_db, source_table="orders_bronze",
#                       sink=_db, sink_table="orders_silver")
#     .drop_nulls(subset=["order_id"])
#     .deduplicate(subset=["order_id"])
# )

# ── Runner ─────────────────────────────────────────────────────────────────────
# The variable MUST be named `runner` — the registry looks for it at load time.
runner = PipelineRunner(db_url=_DB_URL)
# runner.register(_bronze, name="orders_bronze", schedule="0 * * * *")
# runner.register(_silver, name="orders_silver", schedule="15 * * * *",
#                 depends_on="orders_bronze")

# ── Publish to the project registry ──────────────────────────────────────────
# Also discoverable without this line via `get_runner("{slug}")`, since files
# in pipelines/ are auto-loaded. Registering makes it visible to the web UI too.
from tracebi import register

register.pipeline("{slug}", runner)
'''


def _transform_template_text(title: str) -> str:
    today = date.today().isoformat()
    slug = _slugify(title)
    return f'''\
"""
{title}
{'=' * len(title)}

Phase ① — TRANSFORM. Scaffolded by ``tracebi new-transform`` on {today}.

Ordinary, unconstrained pandas: read the raw pull, do whatever the data
needs — window functions, prose parsing, cleaning, dedupe — then SINK the
clean, star-shaped result into the warehouse. The framework does not
constrain this phase; the contract is not *how* you clean, it is *what
lands* — the named tables at the bottom of this file. Phase ② (models/)
reads those tables and never sees this code.

    python transforms/{slug}.py

Keep it idempotent: a rerun replaces the warehouse tables.
"""

from __future__ import annotations

import os

import pandas as pd

from tracebi.connectors.duckdb_connector import DuckDBConnector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAREHOUSE = os.path.join(ROOT, "data", "warehouse.duckdb")


# This file is notebook-shaped: the `# %%` markers below are cell
# boundaries, and `# %% [markdown]` cells are prose. VS Code, Cursor,
# PyCharm, and Jupyter (via jupytext) open it AS a notebook — collapse
# cells, run cell-by-cell, write markdown beside code — while the file
# stays plain, reviewable Python that runs top-to-bottom with
# `python transforms/{slug}.py`. While `tracebi dev` serves, any cell can
# `tracebi.workbench.show(df, note=...)` to put its output in the portal.

# %% [markdown]
# ## Methodology
#
# Narrate the cleaning here — what gets dropped and why, how keys are
# chosen. Distill the load-bearing claims into the sink contract's
# `note=` at the bottom; they travel with the certificate.

# %%  Read the raw input (inputs/, an API pull, a SQL export…)
# df = pd.read_csv(os.path.join(ROOT, "inputs", "orders.csv"))  # ← your raw pull

# %%  Clean — any pandas you like; none of this is traced, by design
# df = df.dropna(subset=["id"]).drop_duplicates(subset=["id"])

# %%  Shape into star-schema tables (facts keyed to dimensions)
# dim_x = df[["x"]].drop_duplicates().rename_axis("x_id").reset_index()
# fact = df.merge(dim_x, on="x")[["id", "x_id", "value"]]

# %%  Sink — the contract: these named tables are what phase ② models
os.makedirs(os.path.dirname(WAREHOUSE), exist_ok=True)
wh = DuckDBConnector("warehouse", database=WAREHOUSE)
# wh.write(dim_x, "dim_x")
# wh.write(fact, "fact_{slug}")

# %%  Declare the sink CONTRACT (optional, recommended)
# Read-only SQL checks on what just landed, recorded beside the warehouse.
# A failed check raises; note= carries your stated methodology into the
# certificate. This certifies the SINK, never the pandas above.
# from tracebi.contracts import contract
# with contract("{slug}", warehouse=WAREHOUSE,
#               note="stated methodology: what was dropped, and why") as c:
#     c.rows("fact_{slug}", at_least=1)
#     c.unique("dim_x", ["x_id"])
#     c.not_null("fact_{slug}", ["id", "value"])
#     c.foreign_key("fact_{slug}", "x_id", refers_to=("dim_x", "x_id"))

# %%
print(f"sunk → {{WAREHOUSE}}")
'''


# ── init project scaffolding ────────────────────────────────────────────────

_INIT_GITIGNORE = _scaffold_text("init_gitignore.txt")

_INIT_ENV_EXAMPLE = _scaffold_text("init_env_example.txt")

_INIT_SAMPLE_CSV = _scaffold_text("init_sample_orders.csv")

_INIT_SAMPLE_TRANSFORM = _scaffold_text("init_sample_transform.py.txt")

_INIT_SAMPLE_MODEL = _scaffold_text("init_sample_model.yaml.txt")

# The source the model reads (connections/warehouse.yaml): where phase ① sinks.
_INIT_CONNECTION = _scaffold_text("init_connection_warehouse.yaml.txt")

# The pipeline every model gets (tracebi.pipeline.model_pipeline): the
# transform, then the model's reports. It is what Refresh data runs in the app.
_INIT_SAMPLE_PIPELINE = _scaffold_text("init_sample_pipeline.yaml.txt")

# The sample report is an ARTIFACT PACKAGE — the one report lane — so the
# first page a new project renders demonstrates the real product: figure
# claims, the presentation stack, provenance badges, and a receipt that
# joins the sink contract. (A JSON spec still renders and `tracebi migrate
# spec` compiles one, but the scaffold must not teach the legacy form.)
_INIT_SAMPLE_REPORT_JSON = _scaffold_text("init_sample_report.json")

_INIT_SAMPLE_TEMPLATE_HTML = _scaffold_text("init_sample_template.html")

# Claude Code and Cursor both read this shape. The command is `tracebi` on
# PATH, run from the project folder — not an absolute path, so the file
# stays the same on every machine.
_INIT_MCP_JSON = json.dumps(
    {"mcpServers": {"tracebi": {"command": "tracebi", "args": ["mcp"]}}},
    indent=2,
) + "\n"

def _init_project_readme(project: str) -> str:
    return f"""\
# {project}

A TraceBi project. Scaffolded by `tracebi init`.

Ask it, build it, schedule it. Analysts and agents write every kind of
report from one set of approved definitions, and the ones worth repeating
run themselves. Every number carries a receipt you can re-check.

```
⓪  INPUT       inputs/       raw pulls land here (API export · CSV · SQL dump)
①  TRANSFORM   transforms/   unconstrained pandas → SINK star tables
                                      ── freeze: data/warehouse.duckdb ──
②  MODEL       models/       a declarative star schema over the sink
                                      ── freeze: the model (the contract) ──
③  REPORT      reports/      artifact packages whose every figure claims a binding
```

## Install

TraceBi is not on PyPI — install it from GitHub:

```bash
pip install "tracebi[analyst] @ git+https://github.com/saltyscott0521/tracebi"
```

For the web UI and REST API as well (in addition to the analyst extras):

```bash
pip install "tracebi[analyst,web] @ git+https://github.com/saltyscott0521/tracebi"
```

(That install carries the API; the built React bundle is not in the repo the
installer builds from, so `/` explains how to build it. Every `/api/...`
route works either way.)

## Run the whole loop now

The scaffold is a complete working example — messy input included:

```bash
tracebi run-pipeline sample_model           # ① clean + sink, then ③ build → output/sample_model/sample_dashboard.html + receipt
tracebi verify output/sample_model/sample_dashboard.html.manifest.json   # every checked section: REPRODUCES
tracebi serve                               # browse it at http://127.0.0.1:8000
```

That `verify` is the point: it re-runs the recorded queries against the model
and confirms the rendered numbers still reproduce. Every number drawn through
the model has a receipt; anything else must be marked as not having one.

**Handing the report to a reviewer?** They need only the `.html` — no model, no
warehouse. The offline check re-hashes the data embedded in the file against
the receipt it carries:

```bash
tracebi verify --file output/sample_model/sample_dashboard.html   # FILE INTACT, or names what was altered
```

## Layout

```
{project}/
├── inputs/           Phase ⓪ — raw pulls (orders.csv is the sample)
├── transforms/       Phase ① — pandas that reads inputs/, sinks star tables
├── models/           Phase ② — each .py exposes `model` (a DataModel)
├── reports/          Phase ③ — one folder per model: reports/<model>/<report>/
├── pipelines/        One per model: transform, then build its reports — each .py exposes `runner`
├── data/             The warehouse (gitignored)
├── output/           Rendered reports; *.manifest.json receipts stay tracked
└── .env.example      Copy to `.env` and fill in credentials
```

Everything in the artifact directories is discovered automatically — by
`tracebi serve`, and by notebooks via `get_model()` / `get_runner()`. There
is no registration file to edit.

Run `git init && git add . && git commit -m init` — manifests stamp the
commit (`git_sha`), which
is half the audit story.

## Wire your own data

1. Copy `.env.example` to `.env` and add your database URL.
2. Drop a raw pull in `inputs/` (or query it in your transform directly).
3. `tracebi new-transform "Orders Clean"` — write the pandas; end by sinking
   named tables to the warehouse. The framework does not constrain this
   phase; the contract is what lands.
4. `tracebi new-model "Sales"` — declare the star schema over those tables.
   `tracebi validate` confirms it loads and its dimension keys are unique.
5. Reports go in a folder named for their model: copy
   `reports/sample_model/sample_dashboard/` to `reports/sales/my_report/` (or
   `tracebi new-report "sales/My Report"`), point its `report.json` bindings
   at your model, and put figures in the template: `data-tb-figure` +
   `data-tb-binding` on any element — spans in prose included.
   `tracebi dev sales/my_report` opens the live loop in the app (edit, watch, leave notes).
   Exploration happens *inside* the artifact — blocks marked
   `data-tb-stage="exploration"` die at the final build.
6. Copy `pipelines/sample_model.py` to `pipelines/sales.py` and name your model
   and transform in it. `tracebi run-pipeline sales` then runs the transform
   and rebuilds the model's reports, and Refresh data in the app runs the same.

## Agents

The same contracts drive the agent surface: `tracebi context` emits the
vocabulary as JSON, `tracebi mcp` opens the gateway (validate a spec, query
a model, render, verify — read-and-compute only). An agent and an analyst
author against the same model and produce the same receipts. **`AGENTS.md`
in this project orients an AI agent working here** — point a fresh session at
it (most coding agents read it automatically).

Your agent's TraceBi tools are already wired (`.mcp.json` for Claude Code,
`.cursor/mcp.json` for Cursor — both run `tracebi mcp` over stdio). Run
`tracebi mcp config --client ...` for other clients.
"""


_INIT_AGENTS_MD = _scaffold_text("init_agents.md")


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> int:
    template = getattr(args, "template", None)
    if template:
        from tracebi.project_templates import init_template_project
        return init_template_project(
            Path(args.project), template, force=bool(args.force),
        )

    target = Path(args.project).resolve()
    if target.exists() and any(target.iterdir()):
        if not args.force:
            print(
                f"refusing to init into non-empty {target}; pass --force to override",
                file=sys.stderr,
            )
            return 1

    # The full three-phase layout: the workflow folders (inputs/, transforms/)
    # plus every directory the server auto-discovers at startup. The scaffold
    # is a complete working example — the first thing a new user runs ends
    # with `tracebi verify` reading REPRODUCES.
    # No requests/ — the exploration story is the artifact's own:
    # `tracebi dev` + exploration blocks that die at build (architecture v2 §7).
    for d in ("inputs", "transforms", "connections", "models", "pipelines",
              "reports", "data", "output"):
        (target / d).mkdir(parents=True, exist_ok=True)

    files = {
        target / ".gitignore":              _INIT_GITIGNORE,
        target / ".env.example":            _INIT_ENV_EXAMPLE,
        target / "README.md":               _init_project_readme(target.name),
        target / "AGENTS.md":               _INIT_AGENTS_MD,
        target / "inputs" / "orders.csv":   _INIT_SAMPLE_CSV,
        target / "transforms" / "sample_transform.py": _INIT_SAMPLE_TRANSFORM,
        target / "connections" / "warehouse.yaml": _INIT_CONNECTION,
        target / "models" / "sample_model.yaml": _INIT_SAMPLE_MODEL,
        target / "pipelines" / "sample_model.yaml": _INIT_SAMPLE_PIPELINE,
        target / "reports" / "sample_model" / "sample_dashboard" / "report.json":
            _INIT_SAMPLE_REPORT_JSON,
        target / "reports" / "sample_model" / "sample_dashboard" / "template.html":
            _INIT_SAMPLE_TEMPLATE_HTML,
        target / ".mcp.json": _INIT_MCP_JSON,
        target / ".cursor" / "mcp.json": _INIT_MCP_JSON,
    }

    for path, content in files.items():
        if path.exists() and not args.force:
            print(f"skipping existing {path}", file=sys.stderr)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Initialised TraceBi project at {target}")
    print(f"  cd {target.name}")
    print(f"  git init && git add . && git commit -m init   # manifests stamp the commit (git_sha)")
    print(f"  tracebi run-pipeline sample_model      # ① clean + sink, then ③ build the report + receipt")
    print(f"  tracebi verify output/sample_model/sample_dashboard.html.manifest.json   # re-run the queries")
    print(f"  tracebi serve                          # browse at http://127.0.0.1:8000")
    print(f"AGENTS.md orients an AI agent working in this project.")
    return 0


def cmd_knowledge(args: argparse.Namespace) -> int:
    """
    The analyst knowledge base — good-practice lessons for authoring analysis.

    ``tracebi knowledge`` lists the curriculum (slug — title — when to reach for
    it); ``tracebi knowledge <slug>`` prints one lesson in full. The same
    lessons back the ``knowledge`` block of ``tracebi context`` and the
    ``tracebi-analyst`` skill, so an agent can pull the relevant one at the
    moment it's making the decision the lesson is about.
    """
    from tracebi.knowledge import get_lesson, list_lessons

    slug = getattr(args, "slug", None)
    if not slug:
        for ls in list_lessons():
            print(f"{ls.slug}\n    {ls.title}")
            if ls.when:
                print(f"    when: {ls.when}")
        print("\nRead one with:  tracebi knowledge <slug>")
        return 0
    lesson = get_lesson(slug)
    if lesson is None:
        print(f"No lesson '{slug}'. Run 'tracebi knowledge' to list them.",
              file=sys.stderr)
        return 1
    print(f"# {lesson.title}\n")
    print(lesson.body)
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """
    Print TraceBi's vocabulary as JSON — every section and panel type with
    its fields and allowed values, the DataSet verbs, measure kinds, filter
    operators, and the discovery conventions.

    Generated from the code, so it cannot drift. Intended as context for a
    tool or agent authoring a project; add --model to include a specific
    model's tables, dimensions and declared measures.
    """
    from tracebi.capabilities import describe

    payload = describe(brief=getattr(args, "brief", False))
    if args.model:
        from tracebi.model_registry import get_model
        try:
            payload["model"] = get_model(args.model).info()
        except (KeyError, AttributeError) as exc:
            print(f"Could not describe model '{args.model}': {exc}", file=sys.stderr)
            return 1

    print(json.dumps(payload, indent=None if args.compact else 2, default=str))
    return 0


def _load_project_models() -> dict:
    """Every model in models/, keyed by the name the spec would reference."""
    from tracebi import model_registry

    models: dict = {}
    d = _default_models_dir()
    if not d.is_dir():
        return models
    for stem in model_registry.auto_discover(str(d)):
        try:
            m = model_registry.get_model(stem)
        except Exception:  # noqa: BLE001 — validate reports load failures itself
            continue
        models[m.name] = m
        models.setdefault(stem, m)
    return models


def cmd_spec(args: argparse.Namespace) -> int:
    """
    Work with report specs — a report as JSON rather than Python.

        tracebi spec schema                 # the JSON Schema
        tracebi spec validate report.json   # check it without running it
        tracebi spec render report.json     # build and render it
    """
    from tracebi.spec import ReportSpec, json_schema

    if args.action == "schema":
        print(json.dumps(json_schema(), indent=2))
        return 0

    if not args.file:
        print(f"`tracebi spec {args.action}` needs a spec file.", file=sys.stderr)
        return 1
    path = Path(args.file)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    # A template-package report.json (a 'data' bindings block, or a template.html
    # sibling) is not a ReportSpec — from_json would reject its 'data'/'libs'
    # keys as "unknown fields", which reads as "your file is malformed" rather
    # than "wrong lane". Name it, and point at the package flow.
    try:
        _raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        _raw = None
    if isinstance(_raw, dict) and (
            "data" in _raw or (path.parent / "template.html").is_file()):
        print(f"{path} is a template-package report.json (data bindings + a "
              f"template.html), not a JSON ReportSpec — `tracebi spec validate` "
              f"only checks the spec lane. Validate a package by building it: "
              f"`tracebi report build {path.parent.name}`.", file=sys.stderr)
        return 1

    try:
        spec = ReportSpec.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad file is a user error
        print(f"{path}: could not read spec — {exc}", file=sys.stderr)
        return 1

    models = _load_project_models()
    result = spec.validate(models)

    for warn in result["warnings"]:
        print(f"· {warn}")
    for err in result["errors"]:
        print(f"✗ {err}", file=sys.stderr)
    if not result["ok"]:
        print(f"\n{len(result['errors'])} problem(s) in {path}.", file=sys.stderr)
        return 1

    if args.action == "validate":
        coverage = spec.data_coverage()
        print(f"✓ {path} is valid — {len(spec.sections)} section(s), "
              f"{coverage['with_data_ref']}/{coverage['total']} data-bearing "
              f"section(s) have a reference")
        return 0

    # render — one report form: compile the spec to the artifact package and
    # render it like a hand-authored package, so it gets figures, badges, the
    # receipt drawer, and a schema-2 manifest instead of the legacy bare HTML.
    import tempfile

    from tracebi.reports.compile_spec import compile_spec
    from tracebi.reports.template_package import TemplatePackage

    compiled = compile_spec(
        spec,
        theme_css=_report_theme_css(spec, extra_theme=getattr(args, "theme", None)),
        script_js=_report_script_js(spec),
    )
    out = Path(args.output or f"{_slugify(spec.name)}.html")
    with tempfile.TemporaryDirectory() as d:
        for fname, content in compiled.files.items():
            (Path(d) / fname).write_text(content, encoding="utf-8")
        TemplatePackage(d).render(models, str(out))
    for warning in compiled.warnings:
        print(f"  · {warning}")
    print(f"Rendered {spec.name} → {out}")
    return 0


def _web_app_importable() -> bool:
    """True when the FastAPI app package can be imported.

    ``tracebi serve`` boots ``tracebi.web.api.main:app``. The wheel ships that
    package, but an install predating the move out of the top-level ``web``
    package does not — checked up front so the failure is an actionable
    message instead of uvicorn's ModuleNotFoundError mid-boot. (The web app is
    not the same thing as the built UI: a wheel can ship the app with no
    bundle, which boots and explains itself at ``/``.)
    """
    import importlib.util

    try:
        return importlib.util.find_spec("tracebi.web.api.main") is not None
    except ImportError:
        return False


def cmd_login(args: argparse.Namespace) -> int:
    """Sign in to a TraceBi server through the browser (OAuth + PKCE)."""
    from tracebi import credentials

    try:
        credentials.login(args.server, timeout=args.timeout)
    except credentials.CredentialsError as exc:
        print(f"tracebi login: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    """Revoke this machine's sign-in to a server and forget it."""
    from tracebi import credentials

    try:
        credentials.logout(args.server)
    except credentials.CredentialsError as exc:
        print(f"tracebi logout: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Is there a newer TraceBi, and how does this install get it?

    Asks GitHub for the latest release (tracebi/_updates.py), prints what's
    new and the one command that updates *this* install. A pip install can
    run it here (asked first, or --yes); a Docker or git-checkout install
    prints it, since those update from outside the process.
    """
    from tracebi import _updates

    if not _updates.enabled():
        print("Update checks are off (TRACEBI_UPDATE_CHECK=0).")
        return 0
    st = _updates.status(force=True)
    print(f"installed: TraceBi {st['current']} ({st['kind']} install)")
    if st["latest"] is None:
        print("latest:    unknown — no published release could be reached "
              "(offline, or none published yet).")
        return 0
    if not st["available"]:
        print(f"latest:    {st['latest']} — you're up to date.")
        return 0
    print(f"latest:    {st['latest']}  {st['url'] or ''}".rstrip())
    notes = [ln for ln in (st.get("notes") or "").splitlines() if ln.strip()]
    if notes:
        print("\nWhat's new:")
        for ln in notes[:15]:
            print(f"  {ln}")
        if len(notes) > 15:
            print(f"  … the rest: {st['url']}")
    print(f"\nTo update:\n  {st['command']}")
    if args.check or st["kind"] != "pip":
        if st["kind"] == "docker":
            print("\nRun it on the host, where deploy/compose.yml lives; a "
                  "container can't replace itself.")
        _update_afterwards()
        return 0
    if not args.yes:
        try:
            answer = input("\nRun it now? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            return 0
    import shlex
    import subprocess
    target = shlex.split(st["command"])[-1]
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", target])
    if rc == 0:
        print(f"\nUpdated to TraceBi {st['latest']}.")
        _update_afterwards()
    return rc


def _update_afterwards() -> None:
    print("\nAfterwards: restart `tracebi serve` (or the container), then run "
          "`tracebi verify output/<report>.html.manifest.json` to confirm your "
          "reports still reproduce on the new version.")


def cmd_serve(args: argparse.Namespace) -> int:
    """
    Serve the current project's web UI.

    The one CLI step between an installed package and a running app.
    Artifacts are discovered from the working directory, so this is run
    from a project root — the layout ``tracebi init`` creates.
    """
    cwd = Path.cwd()
    discovered = {
        d: len([p for p in (cwd / d).iterdir()
                if p.suffix in (".py", *(_MODEL_FILE_SUFFIXES if d == "models" else
                                         _PIPELINE_FILE_SUFFIXES if d == "pipelines" else ()))
                and not p.name.startswith("_")])
        for d in ("models", "pipelines", "reports")
        if (cwd / d).is_dir()
    }
    if not discovered:
        print(
            f"No TraceBi project found in {cwd}.\n"
            f"Expected at least one of: models/ pipelines/ reports/\n"
            f"Run `tracebi init .` to scaffold one, or cd to a project root.",
            file=sys.stderr,
        )
        return 1

    # The app reads artifact directories relative to the working directory,
    # so it must stay on sys.path for the discovery imports to resolve.
    # Inserted before the checks below so a checkout in cwd counts as one.
    sys.path.insert(0, str(cwd))

    if not _web_app_importable():
        print(
            "tracebi serve: the web app (tracebi.web.api) is not importable.\n"
            "Current wheels ship it inside the `tracebi` package; an install\n"
            "predating the move shipped it as a top-level `web` package.\n"
            "Upgrade, or run `tracebi serve` from a clone of the repo:\n"
            "    git clone https://github.com/saltyscott0521/tracebi\n"
            "or point PYTHONPATH at an existing checkout:\n"
            "    PYTHONPATH=/path/to/tracebi-checkout tracebi serve\n"
            "(TraceBi is not on PyPI; the library installs with:\n"
            '    pip install "tracebi[web] @ git+https://github.com/saltyscott0521/tracebi")',
            file=sys.stderr,
        )
        return 1

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "The web UI needs the web extras. TraceBi is not on PyPI —\n"
            "install from the repo:\n"
            '    pip install "tracebi[web] @ git+https://github.com/saltyscott0521/tracebi"',
            file=sys.stderr,
        )
        return 1

    # Don't drag the bundled demo app into someone else's project — it
    # references demo data they do not have. An app module is only needed
    # for connectors; opt in by setting TRACEBI_APP.
    os.environ.setdefault("TRACEBI_APP", "")

    summary = ", ".join(f"{n} {d}" for d, n in discovered.items() if n) or "no artifacts yet"
    print(f"TraceBi — serving {cwd.name} ({summary})")
    print(f"  http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(
        "tracebi.web.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _dev_classic(args: argparse.Namespace) -> bool:
    """Whether ``tracebi dev`` serves the classic preview: asked for with
    ``--classic``, or the web extra the app needs is not installed (said once,
    with the command that adds it)."""
    if args.classic:
        return True
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print("tracebi dev opens the web app, which needs the web extra "
              "(pip install 'tracebi[web]'). Serving the classic preview "
              "instead.", file=sys.stderr)
        return True
    return False


def cmd_dev(args: argparse.Namespace) -> int:
    # The app is the one place you build: an artifact package under reports/
    # opens in it with Build mode on, and no name at all opens it where the
    # agent's exhibits land (before any report exists). --classic is the old
    # stdlib preview server and /__workbench page, kept as the fallback.
    from tracebi._dev_server import serve_dev
    from tracebi.report_paths import open_report
    if args.name is None:
        if _dev_classic(args):
            return serve_dev(None, port=args.port,
                             open_browser=not args.no_browser)
        return _dev_in_the_app(args)
    reports_dir = _default_reports_dir()
    opened = open_report(args.name, purpose="view", reports_dir=reports_dir)
    # A name the guard refuses keeps the historical lookup, so the message
    # below stays the one this command already prints.
    if opened.name_error:
        pkg_dir = reports_dir / args.name
        ready = ((pkg_dir / "report.json").is_file()
                 and (pkg_dir / "template.html").is_file())
    else:
        pkg_dir = opened.path
        ready = opened.package_dir is not None and opened.has_template
    if ready:
        if _dev_classic(args):
            return serve_dev(pkg_dir, port=args.port,
                             open_browser=not args.no_browser)
        return _dev_in_the_app(args)
    print(f"Report package not found: {args.name}. Expected a package at "
          f"{pkg_dir} (report.json + template.html). Scaffold one with "
          f"`tracebi new-report`.", file=sys.stderr)
    return 1


def _dev_in_the_app(args: argparse.Namespace) -> int:
    """``tracebi dev [<name>]``: the web app, in dev mode. With a name it opens
    that report with Build already on; without one, the Reports list, where what
    the agent shows you lands before any report exists. The agent's chat goes
    beside it; the app is where you watch it work and point at what you mean
    (see docs/architecture/design-direction.md).
    """
    from urllib.parse import urlencode

    # Build mode's endpoints write dev-state files, so the server is in dev mode
    # (the same gate as /api/_dev/reload) and stays on loopback.
    os.environ["TRACEBI_DEV_MODE"] = "1"
    url = f"http://127.0.0.1:{args.port}/reports"
    if args.name:
        url += "?" + urlencode({"r": args.name, "build": "1"})
        print(f"\n  TraceBi dev — {args.name} in the app, Build mode")
        print(f"  {url}")
        print("  Put your agent's chat beside it (it connects over MCP: `tracebi mcp`).")
        print("  Click a figure in the report to point at it; tell the agent in chat.\n")
    else:
        print("\n  TraceBi dev — the app, before any report is open")
        print(f"  {url}")
        print("  What your agent shows you while it works appears under Reports, "
              "as From your agent.")
        print("  Any script you run can call tracebi.workbench.show(df, note=...) "
              "while this is up — no env var needed.")
        print("  Put your agent's chat beside it (it connects over MCP: `tracebi mcp`).\n")
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    return cmd_serve(argparse.Namespace(host="127.0.0.1", port=args.port, reload=False))


def cmd_validate(args: argparse.Namespace) -> int:
    """
    Sanity-check the project: layout, then the models themselves.

    Layout checks are cheap file stats. The substantive part is loading each
    model in models/ and running ``DataModel.validate()``, which catches the
    failures that would otherwise surface as wrong numbers — chiefly a
    non-unique dimension key silently inflating every additive measure.
    """
    cwd = Path.cwd()
    problems: list[str] = []
    warnings: list[str] = []
    ok: list[str] = []

    # ── Import every discoverable artifact and report what failed ────────
    # A file that raises on import is skipped at startup with only a
    # warning, so this is where it becomes visible.
    from tracebi.web.discovery import auto_discover, discovery_report

    for label in ("reports", "scheduled"):
        d = cwd / label
        if d.is_dir():
            auto_discover(str(d))
    for entry in discovery_report():
        if entry["status"] == "failed":
            problems.append(
                f"✗ {entry['directory']}/{entry['file']}: {entry['reason']}"
            )
        elif entry.get("warning"):
            warnings.append(
                f"! {entry['directory']}/{entry['file']}: {entry['warning']}"
            )
    n_registered = sum(1 for e in discovery_report() if e["status"] == "registered")
    if n_registered:
        ok.append(f"✓ {n_registered} artifact module(s) imported cleanly")

    env_path = cwd / ".env"
    if env_path.is_file():
        ok.append("✓ .env file found")
    else:
        ok.append("· .env not present (only needed if you load it yourself)")

    # ── Models: load each one and check its declared structure ──────────
    models_dir = args.models_dir
    if not models_dir.is_dir():
        warnings.append(f"· models/ not present at {models_dir}")
    else:
        from tracebi import model_registry

        names = model_registry.auto_discover(str(models_dir))
        if not names:
            warnings.append("· models/ contains no model files")
        for name in names:
            try:
                model = model_registry.get_model(name)
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                ext = next((e for e in ("yaml", "yml", "json")
                            if (models_dir / f"{name}.{e}").is_file()), "py")
                problems.append(f"✗ models/{name}.{ext} failed to load: {exc}")
                continue

            result = model.validate()
            if result["ok"]:
                ndims = len(result["dimensions"])
                ok.append(f"✓ {name}: {model.name} loaded, {ndims} dimension(s) key-unique")
            else:
                for err in result["errors"]:
                    problems.append(f"✗ {name}: {err}")
            for warn in result["warnings"]:
                warnings.append(f"· {name}: {warn}")

    # ── Connections and pipelines declared as YAML: check shape only ─────
    # A connection's ${ENV_VAR}s are resolved when it connects, not here, so a
    # variable that is unset on this machine is not a problem with the file.
    from tracebi.connections import SUFFIXES as _CONN_SUFFIXES, validate_connection_file

    conn_dir = args.connections_dir
    conn_files = sorted(p for p in conn_dir.iterdir()
                        if p.suffix in _CONN_SUFFIXES and not p.name.startswith("_")) \
        if conn_dir.is_dir() else []
    for path in conn_files:
        errors = validate_connection_file(path)
        if errors:
            problems.append(f"✗ {conn_dir.name}/{path.name}: " + "; ".join(errors))
        else:
            ok.append(f"✓ {conn_dir.name}/{path.name} is well formed")
    from tracebi.pipeline.pipeline_spec import validate_pipeline_file

    pipe_dir = args.pipelines_dir
    for path in (sorted(pipe_dir.iterdir()) if pipe_dir.is_dir() else []):
        if path.suffix not in _PIPELINE_FILE_SUFFIXES or path.name.startswith("_"):
            continue
        errors = validate_pipeline_file(path)
        if errors:
            problems.append(f"✗ {pipe_dir.name}/{path.name}: " + "; ".join(errors))
        else:
            ok.append(f"✓ {pipe_dir.name}/{path.name} is well formed")

    for line in ok:
        print(line)
    for line in warnings:
        print(line)
    for line in problems:
        print(line, file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} problem(s) found.", file=sys.stderr)
        return 1
    print("\nProject looks good.")
    return 0


def _project_path(path) -> str:
    """*path* as the person reads it: relative to the project when it is inside
    it (output/sales/q3.html), absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def find_transform(name: str) -> Optional[Path]:
    """The file ``tracebi run-transform <name>`` runs: ``transforms/<name>`` as
    given, or with ``.py`` or ``.ipynb`` added."""
    tdir = Path(os.environ.get("TRACEBI_TRANSFORMS_DIR", "transforms"))
    candidates = [tdir / name, tdir / f"{name}.py", tdir / f"{name}.ipynb"]
    return next((c for c in candidates if c.is_file()), None)


def cmd_run_transform(args: argparse.Namespace) -> int:
    """
    Execute a phase-① transform — ``.py`` or ``.ipynb`` — top-to-bottom in
    a FRESH namespace.

    ``python transforms/<name>.py`` works for scripts already; this verb
    adds the notebook form and, for both, the honesty guarantee that
    matters at the sink: the warehouse comes from a clean top-to-bottom
    execution of the committed file, never from out-of-order kernel state.
    Transforms may be notebook-shaped ``.py`` (``# %%`` cells — every
    notebook editor opens them as notebooks) or literal ``.ipynb``, whose
    code cells are concatenated in order and executed fresh.
    """
    path = find_transform(args.name)
    if path is None:
        tdir = os.environ.get("TRACEBI_TRANSFORMS_DIR", "transforms")
        print(f"transform not found in {tdir}: {args.name}", file=sys.stderr)
        return 1
    print(f"Running {path} (top-to-bottom, fresh namespace)…")
    from tracebi.contracts import ContractViolation
    try:
        if path.suffix == ".ipynb":
            from tracebi._notebook import notebook_to_source
            source = notebook_to_source(path)
            ns: dict = {"__name__": "__main__", "__file__": str(path)}
            exec(compile(source, str(path), "exec"), ns)
        else:
            runpy.run_path(str(path), run_name="__main__")
    except ContractViolation as exc:
        # The sink refused to certify. This is an expected, well-worded
        # failure — print it clean, not as a runpy stack trace (the full
        # trace is behind TRACEBI_DEBUG for anyone debugging the check itself).
        if os.environ.get("TRACEBI_DEBUG"):
            raise
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    from tracebi.connect import connect_command
    return connect_command(args)


def cmd_import_dbt(args: argparse.Namespace) -> int:
    from tracebi.dbt_import import import_dbt_command
    return import_dbt_command(args)


def cmd_new_model(args: argparse.Namespace) -> int:
    if args.from_connection or args.tables:
        from tracebi.connect import draft_model_command
        return draft_model_command(args)
    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(args.title)
    out_path = models_dir / f"{slug}.{'py' if args.python else 'yaml'}"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite existing {out_path}; pass --force to replace",
              file=sys.stderr)
        return 1
    twins = [models_dir / f"{slug}{e}" for e in (".py", *_MODEL_FILE_SUFFIXES)
             if (models_dir / f"{slug}{e}").exists() and models_dir / f"{slug}{e}" != out_path]
    if twins:
        print(f"refusing: {twins[0]} exists; a model is one file, so delete one",
              file=sys.stderr)
        return 1

    if not args.python:
        out_path.write_text(_model_yaml_template_text(args.title), encoding="utf-8")
        print(f"Created {out_path}")
        print(f"  Edit the file; it is picked up by discovery and by:")
        print(f'    from tracebi.model_registry import get_model; get_model("{slug}")')
        return 0

    out_path.write_text(_model_template_text(args.title), encoding="utf-8")
    print(f"Created {out_path}")
    print(f"  Edit the file, then use it with:")
    print(f"    from tracebi.model_registry import get_model")
    print(f'    model = get_model("{slug}")')
    return 0


def cmd_new_transform(args: argparse.Namespace) -> int:
    transforms_dir: Path = args.transforms_dir
    transforms_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(args.title)
    out_path = transforms_dir / f"{slug}.py"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite existing {out_path}; pass --force to replace",
              file=sys.stderr)
        return 1

    out_path.write_text(_transform_template_text(args.title), encoding="utf-8")
    print(f"Created {out_path}")
    print(f"  Write the pandas, end by sinking named tables, then run it:")
    print(f"    python {out_path}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    models_dir: Path = args.models_dir
    if not models_dir.is_dir():
        print(f"No models directory at {models_dir}")
        return 0
    files = sorted(p for p in models_dir.iterdir()
                   if p.suffix in (".py", *_MODEL_FILE_SUFFIXES) and not p.name.startswith("_"))
    if not files:
        print(f"No model files found in {models_dir}")
        return 0
    for p in files:
        print(p.relative_to(models_dir.parent))
    return 0


def _print_warehouse(result: dict) -> None:
    if not result.get("ok", True):
        print(result.get("error", "failed"), file=sys.stderr)
    for match in result.get("columns") or []:
        print(f"{match['connector']}.{match['table']}")
        for col in match.get("columns") or []:
            print(f"  {col['name']}  {col['dtype']}")
    for entry in result.get("connectors") or []:
        print(entry["name"])
        if "error" in entry:
            print(f"  {entry['error']}")
            continue
        tables = entry.get("tables")
        if tables is None:
            print("  (cannot list tables)")
        elif not tables:
            print("  (no tables)")
        else:
            for name in tables:
                print(f"  {name}")


def cmd_warehouse(args: argparse.Namespace) -> int:
    from tracebi.mcp_server import gateway_describe_table

    result = gateway_describe_table(
        table=args.table or "",
        connector=args.connector or "",
    )
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_warehouse(result)
    return 0 if result.get("ok", True) else 1


def cmd_new_pipeline(args: argparse.Namespace) -> int:
    pipelines_dir: Path = args.pipelines_dir
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(args.title)
    out_path = pipelines_dir / f"{slug}.{'py' if args.python else 'yaml'}"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite existing {out_path}; pass --force to replace",
              file=sys.stderr)
        return 1
    twins = [pipelines_dir / f"{slug}{e}" for e in (".py", *_PIPELINE_FILE_SUFFIXES)
             if (pipelines_dir / f"{slug}{e}").exists() and pipelines_dir / f"{slug}{e}" != out_path]
    if twins:
        print(f"refusing: {twins[0]} exists; a pipeline is one file, so delete one",
              file=sys.stderr)
        return 1

    if not args.python:
        out_path.write_text(_pipeline_yaml_template_text(args.title), encoding="utf-8")
        print(f"Created {out_path}")
        print(f"  Name its transform and models, then run it with:")
        print(f"    tracebi run-pipeline {slug}")
        return 0

    out_path.write_text(_pipeline_template_text(args.title), encoding="utf-8")
    print(f"Created {out_path}")
    print(f"  Edit the file, then use it with:")
    print(f"    from tracebi.pipeline_registry import get_runner")
    print(f'    runner = get_runner("{slug}")')
    return 0


def _pipeline_plan(runner) -> list[str]:
    """
    Every registered layer, upstream-first, each appearing once.

    ``execution_order`` resolves the chain for a single layer. Concatenating
    each layer's chain and keeping the first occurrence gives a valid order
    for the whole pipeline: within any chain a layer's upstreams precede it,
    and deduplicating on first sight can only move a layer earlier.
    """
    plan: list[str] = []
    for layer in runner.layers():
        for step in runner.execution_order(layer["name"]):
            if step not in plan:
                plan.append(step)
    return plan


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    """
    Run a pipeline from the command line.

    This is the execution plane's entry point. Until it existed the only way
    to execute a layer was ``POST /api/pipelines/{name}/layers/{layer}/run``,
    which meant the web server had to be running — and had to be the thing
    doing the work — for any data to be produced. That is what tied batch
    execution to the serving process and made deployments where the two are
    separate (a scheduled job writing to Postgres, a stateless API reading
    from it) impossible to express. See NOTES.md, "Deployment planes".

    Any external scheduler can now drive this: cron, a Kubernetes CronJob,
    Airflow, a CI job. TraceBi does not need to own the schedule.
    """
    import getpass

    from tracebi.audit import set_actor
    from tracebi.pipeline import run_record
    from tracebi.pipeline_registry import get_runner, list_pipelines

    # Attribute CLI-driven runs too. A cron job or CI step runs as some
    # account, and "who ran this" should answer that rather than shrug at
    # everything the web UI did not trigger.
    try:
        set_actor(getpass.getuser(), role="cli")
    except Exception:  # noqa: BLE001 — no controlling user (some containers)
        set_actor(None, role="cli")

    try:
        runner = get_runner(args.name)
    except KeyError as exc:
        print(str(exc).strip("\"'"), file=sys.stderr)
        known = list_pipelines()
        if not known:
            print(f"No pipelines found in {args.pipelines_dir}. "
                  'Scaffold one with: tracebi new-pipeline "My ETL"', file=sys.stderr)
        return 1

    if args.status:
        rows = runner.layers()
        if not rows:
            print(f"Pipeline '{args.name}' has no registered layers.")
            return 0
        width = max(len(r["name"]) for r in rows)
        for r in rows:
            # layers() returns registration data only — name, type, schedule,
            # depends_on. Run status is a separate DB read per layer, which is
            # what the web router does too.
            try:
                last = runner.last_run(r["name"])
            except Exception as exc:  # noqa: BLE001
                status, when = f"error: {type(exc).__name__}", ""
            else:
                status = (last or {}).get("status") or "never run"
                when = (last or {}).get("completed_at") or (last or {}).get("started_at") or ""
            print(f"  {r['name']:<{width}}  {r['type']:<12} {status:<10} {when}")
        return 0

    if args.layer:
        if not runner.has_layer(args.layer):
            print(f"Layer '{args.layer}' is not registered in pipeline '{args.name}'. "
                  f"Available: {[r['name'] for r in runner.layers()]}", file=sys.stderr)
            return 1
        chain = runner.execution_order(args.layer) if args.refresh else [args.layer]
    else:
        chain = _pipeline_plan(runner)

    if not chain:
        print(f"Pipeline '{args.name}' has no registered layers.")
        return 0

    print(f"[tracebi] {args.name}: {' → '.join(chain)}")
    # Recorded like a run started from the app: a row in the state store and a
    # log, so the model's History lists both. Recording is not running, though: a
    # store that cannot be reached costs the record, not the refresh.
    try:
        run_id = run_record.begin(args.name, chain)
    except Exception as exc:  # noqa: BLE001
        print(f"[tracebi] this run is not being recorded: {exc}", file=sys.stderr)
        run_id = None
    failed: list[str] = []
    with run_record.recording(args.name, run_id, chain) as rec:
        for step in chain:
            try:
                runner.execute_layer(step)
            except Exception as exc:  # noqa: BLE001 — report and keep going
                # Downstream layers read what upstream wrote, so a failure part
                # way through leaves the rest resting on stale data. Report every
                # failure rather than stopping at the first, and exit non-zero so
                # whatever scheduler invoked this can act on it.
                failed.append(step)
                rec.fail(exc)
                print(f"[tracebi] {step} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

        if failed:
            print(f"[tracebi] {len(failed)} of {len(chain)} layer(s) failed: "
                  f"{', '.join(failed)}", file=sys.stderr)
    if failed:
        return 1

    print(f"[tracebi] {len(chain)} layer(s) completed.")
    return 0


def cmd_list_pipelines(args: argparse.Namespace) -> int:
    pipelines_dir: Path = args.pipelines_dir
    if not pipelines_dir.is_dir():
        print(f"No pipelines directory at {pipelines_dir}")
        return 0
    files = sorted(p for p in pipelines_dir.iterdir()
                   if p.suffix in (".py", *_PIPELINE_FILE_SUFFIXES) and not p.name.startswith("_"))
    if not files:
        print(f"No pipeline files found in {pipelines_dir}")
        return 0
    for p in files:
        print(p.relative_to(pipelines_dir.parent))
    return 0


_MCP_CLIENTS = ("claude-code", "cursor", "claude-desktop")
# Placeholders, never values. A printed snippet must not echo
# TRACEBI_MCP_TOKEN even when the variable is set in this process.
# Claude Code's .mcp.json interpolates ${NAME}; Cursor's mcp.json
# interpolates ${env:NAME}.
_MCP_HTTP_AUTH = {
    "claude-code": "Bearer ${TRACEBI_MCP_TOKEN}",
    "cursor": "Bearer ${env:TRACEBI_MCP_TOKEN}",
}
_CLAUDE_DESKTOP_HTTP_REFUSAL = (
    "Add the URL as a custom connector in Claude Desktop "
    "(Settings → Connectors)."
)


def _claude_desktop_stdio() -> dict:
    """Claude Desktop needs an absolute executable. ``which`` first; if
    ``tracebi`` is not on PATH, ``sys.executable -m tracebi.cli``."""
    found = shutil.which("tracebi")
    if found:
        return {"command": str(Path(found).resolve()), "args": ["mcp"]}
    return {"command": sys.executable, "args": ["-m", "tracebi.cli", "mcp"]}


def mcp_client_config(client: str, http_url: Optional[str] = None) -> dict:
    """The snippet ``tracebi mcp config`` prints. stdio for a local agent;
    ``http_url`` is the streamable-HTTP form, with a header placeholder.

    Claude Desktop's config file only starts local stdio servers, so a
    remote URL is refused (see ``cmd_mcp``).
    """
    if http_url:
        if client not in _MCP_HTTP_AUTH:
            raise ValueError(_CLAUDE_DESKTOP_HTTP_REFUSAL)
        server: dict = {
            "type": "http",
            "url": http_url,
            "headers": {"Authorization": _MCP_HTTP_AUTH[client]},
        }
    elif client == "claude-desktop":
        server = _claude_desktop_stdio()
    else:
        server = {"command": "tracebi", "args": ["mcp"]}
    return {"mcpServers": {"tracebi": server}}


def cmd_mcp(args: argparse.Namespace) -> int:
    """
    Serve the agent gateway over the Model Context Protocol.

    stdio by default, which is what a locally-configured agent speaks;
    --transport http for a remote one. Models come from the same project
    conventions as every other command (./models, or TRACEBI_MODELS_DIR).

    The http transport refuses to start until an auth decision is made:
    set TRACEBI_MCP_TOKEN (bearer auth) or pass --insecure explicitly.

    ``tracebi mcp config`` prints a client snippet and does not serve.
    """
    if getattr(args, "mcp_action", None) == "config":
        if args.client not in _MCP_CLIENTS:
            print(
                "tracebi mcp config needs --client "
                "claude-code|cursor|claude-desktop",
                file=sys.stderr,
            )
            return 1
        if args.http and args.client == "claude-desktop":
            print(_CLAUDE_DESKTOP_HTTP_REFUSAL, file=sys.stderr)
            return 1
        print(json.dumps(mcp_client_config(args.client, args.http), indent=2))
        return 0
    if args.client or args.http:
        print("--client and --http belong to `tracebi mcp config`",
              file=sys.stderr)
        return 1
    from tracebi.mcp_server import GatewayAuthError, serve

    try:
        serve(transport=args.transport, port=args.port, host=args.host,
              insecure=args.insecure,
              allow_insecure_bind=args.allow_insecure_bind)
    except (ImportError, GatewayAuthError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


def cmd_verify_file(args: argparse.Namespace) -> int:
    """
    Offline check of a self-contained report ``.html`` (architecture §3.2).

    Recovers the exact fingerprinted bytes from every embedded data block and
    rehashes them — no model, no DataFrame rebuild — against a sibling
    ``<file>.manifest.json``. Catches a number edited in the shipped file, which
    ``verify <manifest>`` (query → model) cannot see. Exit 0 only when every
    embedded block matches the manifest.
    """
    from tracebi.verify import (
        FIGURE_MATCHES, FIGURE_UNVERIFIED_MARK, FILE_FIGURE_LABELS,
        FILE_MATCHES, FILE_STATUS_LABELS, REFUSED_SNAPSHOT, verify_file,
    )

    html_path = Path(args.verify_file)
    if not html_path.is_file():
        print(f"report file not found: {html_path}", file=sys.stderr)
        return 1

    manifest_path = (
        Path(args.file_manifest) if getattr(args, "file_manifest", None)
        else html_path.with_name(html_path.name + ".manifest.json")
    )
    if not manifest_path.is_file():
        print(f"no manifest found next to the report: expected {manifest_path}\n"
              "(pass --manifest <path> to point at it explicitly)",
              file=sys.stderr)
        return 1

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"manifest is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print(f"manifest must be a JSON object, got {type(manifest).__name__}",
              file=sys.stderr)
        return 1

    html = html_path.read_text(encoding="utf-8")
    result = verify_file(html, manifest)

    # A refused snapshot prints only the refusal — a header and "0 bindings
    # checked" would read like a check that ran and found little, and this
    # one deliberately did not run at all.
    if result["verdict"] == REFUSED_SNAPSHOT:
        print(result["verdict_detail"], file=sys.stderr)
        return result["exit_code"]

    name = result["report_name"] or html_path.name
    print(f"Verifying embedded data in '{name}' ({html_path})")
    print(f"  against manifest {manifest_path}")
    width = max(len(v) for v in FILE_STATUS_LABELS.values())
    for b in result["bindings"]:
        status = b["status"]
        mark = "✓" if status == FILE_MATCHES else "✗"
        line = f"{mark} {FILE_STATUS_LABELS[status]:<{width}}  {b['binding']}"
        if status != FILE_MATCHES:
            line += f" — {b['detail']}"
        print(line, file=sys.stdout if mark != "✗" else sys.stderr)

    # Figure cross-check rows (manifest schema 2) — the claims layer, so a
    # failure names the figure, not just the verdict.
    if result.get("figures"):
        fwidth = max(len(v) for v in FILE_FIGURE_LABELS.values())
        print()
        for f in result["figures"]:
            status = f["status"]
            ok = status in (FIGURE_MATCHES, FIGURE_UNVERIFIED_MARK)
            mark = "✓" if status == FIGURE_MATCHES else "·" if ok else "✗"
            line = (f"{mark} {FILE_FIGURE_LABELS[status]:<{fwidth}}  "
                    f"{f.get('figure')}")
            if not ok:
                line += f" — {f['detail']}"
            print(line, file=sys.stdout if mark != "✗" else sys.stderr)

    counts = ", ".join(
        f"{n} {FILE_STATUS_LABELS[status].lower()}"
        for status, n in result["summary"].items() if n
    ) or "no embedded data blocks"
    print(f"\n{len(result['bindings'])} binding(s) checked: {counts}")
    print(result["verdict_detail"],
          file=sys.stdout if result["exit_code"] == 0 else sys.stderr)
    return result["exit_code"]


def cmd_verify(args: argparse.Namespace) -> int:
    """
    Re-verify a rendered manifest: the other half of the stamp.

    Every recorded query in the manifest is re-run against the project's
    models and classified — REPRODUCES, SOURCE DRIFT (result differs and
    an input fingerprint moved), MODEL CHANGED (a table now loads from a
    different source/connector — a governance event), UNEXPLAINED (result
    differs but the inputs did not — the alarming case), or UNVERIFIABLE
    (no recorded query to re-run). One line per section, then a summary
    and the receipt-level verdict.

    Exit codes: 0 all reproduce/unverifiable · 2 diagnosed drift only ·
    1 anything unexplained, of unknown cause, errored, or a manifest with
    no data-bearing section at all (nothing was verified, so nothing passed).
    """
    from tracebi.verify import (
        FIGURE_STATUS_LABELS, REPRODUCES, STATUS_LABELS, UNVERIFIABLE,
        UNVERIFIED, load_models, verify_manifest,
    )

    # Two distinct checks under one verb (architecture §4). `--file` opens the
    # shipped .html and rehashes its embedded bytes against a sibling manifest
    # (embedded bytes → manifest); it never touches a model. The positional
    # `manifest` re-runs recorded queries (query → model). Neither implies the
    # other, so they are kept apart rather than merged.
    if getattr(args, "verify_file", None) is not None:
        return cmd_verify_file(args)
    if not args.manifest:
        print("verify: provide a manifest path, or --file <report.html>",
              file=sys.stderr)
        return 1

    path = Path(args.manifest)
    if not path.is_file():
        print(f"manifest not found: {path}", file=sys.stderr)
        return 1
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"manifest is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print(f"manifest must be a JSON object, got {type(manifest).__name__}",
              file=sys.stderr)
        return 1

    models = load_models(args.verify_models_dir or args.models_dir)
    try:
        result = verify_manifest(manifest, models,
                                 strict=getattr(args, "strict", False))
    except Exception as exc:  # noqa: BLE001 — a corrupt receipt is a user error
        print(f"manifest could not be verified: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        return result["exit_code"]

    name = result["report_name"] or path.name
    print(f"Verifying '{name}' ({path})")
    width = max(len(v) for v in STATUS_LABELS.values())
    for s in result["sections"]:
        status = s["status"]
        mark = ("✓" if status == REPRODUCES
                else "·" if status == UNVERIFIABLE
                else "✗")
        line = f"{mark} {STATUS_LABELS[status]:<{width}}  {s['section']}"
        if status != REPRODUCES:
            line += f" — {s['detail']}"
        print(line, file=sys.stdout if mark != "✗" else sys.stderr)

    counts = ", ".join(
        f"{n} {STATUS_LABELS[status].lower()}"
        for status, n in result["summary"].items() if n
    ) or "no data-bearing sections"
    print(f"\n{len(result['sections'])} section(s) checked: {counts}")

    # Figure rollup (manifest schema 2): the claims layer, figures first.
    if result.get("figures"):
        fwidth = max(len(v) for v in FIGURE_STATUS_LABELS.values())
        print()
        for f in result["figures"]:
            status = f["status"]
            mark = ("✓" if status == REPRODUCES
                    else "·" if status in (UNVERIFIABLE, UNVERIFIED)
                    else "✗")
            line = (f"{mark} {FIGURE_STATUS_LABELS[status]:<{fwidth}}  "
                    f"{f['figure']}")
            if status != REPRODUCES:
                line += f" — {f['detail']}"
            print(line, file=sys.stdout if mark != "✗" else sys.stderr)

    exit_code = result["exit_code"]

    # The phase-① claim, reported beside the figure claims — never blended
    # (v2 §2.6). The recorded block says what the warehouse certified about
    # itself at BUILD time; it is informational here and moves no exit code.
    recorded_contracts = manifest.get("transform_contracts")
    if recorded_contracts:
        print("\nsink contracts (recorded at build):")
        for table, rec in sorted(recorded_contracts.items()):
            status = rec.get("status", "?")
            mark = "✓" if status == "satisfied" else "·"
            line = f"{mark} {status:<12} {table}"
            if status == "satisfied":
                line += f" — {rec.get('checks', 0)} check(s), transform '{rec.get('transform')}'"
            elif status == "stale":
                line += (f" — re-sunk after transform '{rec.get('transform')}' "
                         f"checked it; the certificate no longer describes "
                         f"this data")
            print(line)

    # --contracts: re-run the recorded checks against the CURRENT warehouse.
    # This is its own claim with its own exit: a check failing NOW means the
    # sink no longer satisfies its declared contract. It never says the
    # transform was verified, and it never colors a figure status.
    if getattr(args, "contracts", False):
        from tracebi.contracts import check_fingerprints, rerun_checks
        warehouses = sorted({
            c.database for m in models.values() for c in m.connectors()
            if isinstance(getattr(c, "database", None), str)
            and c.database != ":memory:"
            and type(c).__name__ == "DuckDBConnector"
        })
        rows = [r for wh in warehouses for r in rerun_checks(wh)]
        print("\nsink contracts (re-run now):")
        if not rows:
            print("· no contract record found beside the warehouse")
        failed_now = 0
        for r in rows:
            ok = r.get("passed_now")
            mark = "✓" if ok else "✗"
            desc = f"{r['check']}({r['table']}, {r.get('params')})"
            line = f"{mark} {'satisfied' if ok else 'VIOLATED':<12} {r['transform']}: {desc}"
            if not ok:
                line += f" — observed {r.get('observed_now', r.get('note'))}"
                failed_now += 1
            print(line, file=sys.stdout if ok else sys.stderr)
        if failed_now:
            print(f"{failed_now} check(s) the sink no longer satisfies.",
                  file=sys.stderr)
            exit_code = 1

        # Re-running the checks can miss out-of-band mutation: a row inserted
        # straight into the warehouse that still satisfies every declared
        # check. Compare the certified per-table fingerprints to the data
        # right now — drift means the certificate no longer describes the
        # warehouse (a re-sink would have rewritten it).
        fp_rows = [r for wh in warehouses for r in check_fingerprints(wh)]
        if fp_rows:
            print("\nsink data fingerprints (vs the certificate):")
            drifted = 0
            for r in fp_rows:
                ok = r["matches"]
                state = "unchanged" if ok else (
                    "MISSING" if r["current"] is None else "CHANGED")
                print(f"{'✓' if ok else '✗'} {state:<12} "
                      f"{r['transform']}: {r['table']}",
                      file=sys.stdout if ok else sys.stderr)
                if not ok:
                    drifted += 1
            if drifted:
                print(f"{drifted} table(s) changed since the sink certified "
                      f"them — the certificate no longer describes this "
                      f"warehouse.", file=sys.stderr)
                exit_code = 1

    # Routed by exit code, not by verdict: a run that exits 0 must write
    # nothing to stderr, or CI wrappers that treat stderr as failure will
    # fail a receipt this command just called fine.
    print(result["verdict_detail"],
          file=sys.stdout if exit_code == 0 else sys.stderr)
    return exit_code


def _migrate_model(args: argparse.Namespace) -> int:
    """``tracebi migrate model models/<name>.py [--write]``: the loaded model
    as YAML. Never renames or deletes the Python file: while it exists it wins
    over the YAML, and removing it is the user's step."""
    from tracebi.model.model_yaml import ModelMigrationError, model_to_yaml
    from tracebi.model_registry import ModelRegistry

    src = Path(args.path)
    if src.suffix != ".py" or not src.is_file():
        print(f"expected an existing models/<name>.py, got {src}", file=sys.stderr)
        return 1
    registry = ModelRegistry()
    registry.auto_discover(str(src.parent))
    try:
        model = registry.get(src.stem)
        text = model_to_yaml(model, src.stem, src.resolve().parent.parent)
    except ModelMigrationError as exc:
        print(f"cannot migrate {src}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — a model that will not load is a user error
        print(f"cannot load {src}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        registry.release_all()
    if not args.write:
        print(text, end="")
        return 0
    target = src.with_suffix(".yaml")
    if target.exists() and not args.force:
        print(f"target already exists: {target}\nPass --force to overwrite it.",
              file=sys.stderr)
        return 1
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target}")
    print(f"{src.name} still wins over it (a model is one file): delete {src.name} "
          f"when you are ready to cut over; nothing was renamed or removed.")
    return 0


def _migrate_declarative(args: argparse.Namespace) -> int:
    """``tracebi migrate connection|pipeline <file.py> [--write]``: the loaded
    object as YAML, or a refusal naming the construct that cannot be written
    declaratively. Never deletes the Python file."""
    src = Path(args.path)
    if src.suffix != ".py" or not src.is_file():
        print(f"expected an existing {args.what} .py file, got {src}", file=sys.stderr)
        return 1
    try:
        if args.what == "connection":
            from tracebi.connections import connection_to_yaml

            text = connection_to_yaml(src)
            target = args.connections_dir / f"{src.stem}.yaml"
        else:
            from tracebi.pipeline.pipeline_spec import pipeline_to_yaml
            from tracebi.pipeline_registry import PipelineRegistry

            registry = PipelineRegistry()
            registry.auto_discover(str(src.parent))
            text = pipeline_to_yaml(registry.get(src.stem), src.stem)
            target = src.with_suffix(".yaml")
    except Exception as exc:  # noqa: BLE001 — the reason is the answer
        print(f"cannot migrate {src}: {exc}", file=sys.stderr)
        return 1
    if not args.write:
        print(text, end="")
        return 0
    if target.exists() and not args.force:
        print(f"target already exists: {target}\nPass --force to overwrite it.",
              file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target}")
    print(f"{src.name} still wins over it for a pipeline (one file each): delete "
          f"{src.name} when you are ready to cut over; nothing was removed."
          if args.what == "pipeline" else
          f"{src}: nothing was removed. A model that names this connection as "
          f"`connection: {src.stem}` now reads {target.name} first.")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """
    ``tracebi migrate spec <file.json>`` — compile a JSON spec into an
    artifact package directory beside it.

    Emits alongside, never replaces: ``reports/sales.json`` compiles to
    ``reports/sales/``, sharing the stem — and at discovery the artifact
    directory *shadows* the same-named spec (with a warning naming both),
    so the migration is a cutover the moment the directory exists, and a
    rollback is deleting it. The spec's ``theme``/``script`` files compile
    into the package's ``style.css``/``script.js``.
    """
    if args.what == "model":
        return _migrate_model(args)
    if args.what in ("connection", "pipeline"):
        return _migrate_declarative(args)

    from tracebi.reports.compile_spec import compile_spec
    from tracebi.spec import ReportSpec

    src = Path(args.path)
    if not src.is_file():
        print(f"spec not found: {src}", file=sys.stderr)
        return 1
    try:
        spec = ReportSpec.from_json(src.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad spec is a user error
        print(f"cannot read spec: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # theme/script are filenames resolved against the reports directory —
    # the directory the spec itself lives in.
    theme_css, script_js = "", ""
    for field_name, value in (("theme", spec.theme), ("script", spec.script)):
        if not value:
            continue
        ref = src.parent / value
        if ref.is_file():
            content = ref.read_text(encoding="utf-8")
            if field_name == "theme":
                theme_css = content
            else:
                script_js = content
        else:
            print(f"warning: {field_name} file '{value}' not found beside the "
                  f"spec — not compiled in", file=sys.stderr)

    compiled = compile_spec(spec, theme_css=theme_css, script_js=script_js)

    target = src.parent / src.stem
    if target.exists() and not args.force:
        print(f"target already exists: {target}\n"
              f"Pass --force to overwrite it.", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)
    for fname, content in compiled.files.items():
        (target / fname).write_text(content, encoding="utf-8")

    print(f"Compiled {src.name} → {target}/")
    for fname in compiled.files:
        print(f"  {fname}")
    for w in compiled.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"\nThe artifact now shadows the spec at discovery (delete the "
          f"directory to roll back). Next:\n"
          f"  tracebi report build {src.stem}\n"
          f"  tracebi dev {src.stem}")
    return 0


# ── Report generator: new-report / report build|preview ────────────────────

def _scaffold_binding() -> tuple[str, dict, str]:
    """A concrete ``(model_name, query_dict, note)`` for a starter package.

    So ``tracebi new-report`` renders out of the box, the query is derived from
    the first model in ``models/``: its first fact, first measure, and — when
    the model has one — a dimension attribute to slice by. With no model yet,
    a placeholder is emitted and the note tells the author to point it at one.
    """
    models = _load_project_models()
    for name, model in sorted(models.items()):
        try:
            info = model.info()
        except Exception:  # noqa: BLE001 — an unloadable model is not a scaffold source
            continue
        facts = info.get("facts") or []
        measures = info.get("measures") or []
        if not facts or not measures:
            continue
        query: dict = {"fact": facts[0]["name"],
                       "measures": [measures[0]["name"]]}
        dims = info.get("dimensions") or []
        for d in dims:
            if d.get("attributes"):
                query["dimensions"] = [f"{d['name']}.{d['attributes'][0]}"]
                break
        return info.get("name", name), query, ""
    return (
        "your_model",
        {"fact": "your_fact", "measures": ["your_measure"],
         "dimensions": ["your_dimension.attribute"]},
        "No model found in models/ — edit report.json's 'data' block to name a "
        "real model and query before running `tracebi report build`.",
    )


def _humanise_label(ref: str) -> str:
    """"dim_region.region" → "Region"; "revenue" → "Revenue" — a readable label
    for a scaffolded heading, mirroring the runtime's humanise."""
    name = str(ref).split(".")[-1]
    name = re.sub(r"^(dim|fact)_", "", name).replace("_", " ").strip()
    return name[:1].upper() + name[1:] if name else str(ref)


def _report_json_text(title: str, model: str, query: dict) -> str:
    """Teaching bindings from the derived query: ``totals`` (measures only,
    one row — for KPI cards and bound prose) and, when the model has a
    dimension, ``top`` (the leading member, one row — for the answer-first
    lede) and ``breakdown`` (measures by that dimension, sorted — for a chart
    and a filterable table). The generated template references all of them,
    so the scaffold demonstrates the whole figure grammar rather than a bare
    table."""
    fact = query["fact"]
    measures = query["measures"]
    dims = query.get("dimensions")
    data = {"totals": {"model": model,
                       "query": {"fact": fact, "measures": measures}}}
    if dims:
        data["top"] = {"model": model, "query": {
            "fact": fact, "measures": measures[:1], "dimensions": dims,
            "order_by": ["-" + measures[0]], "limit": 1}}
        data["breakdown"] = {"model": model, "query": {
            "fact": fact, "measures": measures, "dimensions": dims,
            "order_by": ["-" + measures[0]]}}
    declaration = {
        "name": title,
        "author": "",
        "description": "Freeform report package scaffolded by tracebi "
                       "new-report. Every figure claims a stamped binding; edit "
                       "template.html to reshape the page — nothing re-runs the "
                       "pandas.",
        "data": data,
    }
    return json.dumps(declaration, indent=2) + "\n"


def _layout_brief_body(measure: str, dim_ref: "str | None", table_binding: str,
                       esc) -> list[str]:
    """The ``brief`` recipe below the shared header: one chart card.

    With no dimension to chart, the one card is the totals table — the same
    fallback the dashboard uses, and still not a ``.tb-cols-2`` row.
    """
    if not dim_ref:
        return [
            "",
            '  <div class="tb-card">',
            "    <h2>Detail</h2>",
            f'    <table data-tb-figure="table" data-tb-binding="{table_binding}"',
            '           class="tb-table--striped" id="tbl-detail"></table>',
            "  </div>",
        ]
    d_label = _humanise_label(dim_ref)
    m_label = _humanise_label(measure)
    return [
        "",
        '  <div class="tb-card">',
        f"    <h2>{esc(m_label)} by {esc(d_label)}</h2>",
        f'    <div data-tb-figure="chart" data-tb-binding="breakdown"',
        f'         data-tb-type="bar" data-tb-x="{esc(dim_ref)}"',
        f'         data-tb-y="{esc(measure)}" data-tb-value-format="compact"',
        '         id="chart-breakdown"></div>',
        "  </div>",
    ]


def _layout_tabbed_body(measure: str, dim_ref: "str | None", table_binding: str,
                        esc) -> list[str]:
    """The ``tabbed`` recipe: Overview and Detail, one visible at a time."""
    if not dim_ref:
        return [
            "",
            '  <div class="tb-tabs">',
            '    <section data-tb-tab="Overview">',
            '      <p class="tb-note">Add a dimension to the breakdown query',
            "        and the chart for this page goes here.</p>",
            "    </section>",
            '    <section data-tb-tab="Detail">',
            '      <div class="tb-card">',
            "        <h2>Detail</h2>",
            f'        <table data-tb-figure="table" data-tb-binding="{table_binding}"',
            '               class="tb-table--striped" id="tbl-detail"></table>',
            "      </div>",
            "    </section>",
            "  </div>",
        ]
    d_label = _humanise_label(dim_ref)
    m_label = _humanise_label(measure)
    return [
        "",
        '  <div class="tb-tabs">',
        '    <section data-tb-tab="Overview">',
        '      <div class="tb-card">',
        f"        <h2>{esc(m_label)} by {esc(d_label)}</h2>",
        f'        <div data-tb-figure="chart" data-tb-binding="breakdown"',
        f'             data-tb-type="bar" data-tb-x="{esc(dim_ref)}"',
        f'             data-tb-y="{esc(measure)}" data-tb-value-format="compact"',
        '             id="chart-breakdown"></div>',
        "      </div>",
        "    </section>",
        '    <section data-tb-tab="Detail">',
        '      <div class="tb-card">',
        "        <h2>Detail</h2>",
        '        <p class="tb-note">Filter and search subset which stamped rows',
        "          display — they never compute new numbers. The CSV button",
        "          exports the stamped bytes verbatim.</p>",
        "        <p>",
        "          <label>Search",
        '            <input data-tb-search data-tb-binding="breakdown"',
        '                   placeholder="type to filter…"></label>',
        f"          <label>{esc(d_label)}",
        f'            <select data-tb-filter data-tb-binding="breakdown"',
        f'                    data-tb-column="{esc(dim_ref)}"></select></label>',
        '          <button data-tb-download data-tb-binding="breakdown"',
        '                  data-tb-label="Download CSV"></button>',
        "        </p>",
        f'        <table data-tb-figure="table" data-tb-binding="{table_binding}"',
        '               class="tb-table--striped" id="tbl-detail"></table>',
        "      </div>",
        "    </section>",
        "  </div>",
    ]


def _report_template_html(title: str, measure: str, dim_ref: "str | None",
                          layout: str = "dashboard") -> str:
    """A starter ``template.html`` for one named page structure.

    ``dashboard`` (the default, and what ``tracebi init`` writes) is an answer
    sentence, a KPI strip, and — when the model has a dimension — a chart
    beside a filter/search/download table (``.tb-cols-2``), plus a methodology
    block and an exploration stage. ``brief`` is that header and one chart
    card. ``tabbed`` is that header, then Overview and Detail tabs. Generated
    (not a static file) because ``data-tb-*`` attributes name the model's real
    measure and dimension. No ``script.js`` or ``style.css``: the runtime
    (``tracebi.js`` + ``tracebi.css``) draws every figure from the stamped
    bytes — hand-rolling a CSV parser and a chart is exactly the L0 trap a
    scaffold must not teach.
    """
    if layout not in ("brief", "dashboard", "tabbed"):
        raise ValueError(
            f"unknown layout {layout!r}; choose brief, dashboard, or tabbed"
        )
    esc = _html.escape
    m_label = _humanise_label(measure)
    table_binding = "breakdown" if dim_ref else "totals"
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        f"<head><meta charset=\"utf-8\"><title>{esc(title)}</title></head>",
        "<body>",
        '<main class="tb-page">',
        "  <header>",
        f"    <h1>{esc(title)}</h1>",
    ]
    if dim_ref:
        # Lead with the answer: one sentence, numbers bound, under the title.
        parts += [
            '    <p class="tb-lede">',
            '      <span data-tb-figure="value" data-tb-binding="top"',
            f'            data-tb-cell="{esc(dim_ref)}" id="val-top">—</span>',
            f"      leads on {esc(m_label.lower())} with",
            '      <span data-tb-figure="value" data-tb-binding="top"',
            f'            data-tb-cell="{esc(measure)}" id="val-top-{esc(measure)}">—</span>.',
            "      Rewrite this sentence as the page's finding.</p>",
        ]
    parts += [
        '    <p class="tb-note">Every number on this page is a live, fingerprinted',
        f"      query — including this one: {esc(m_label)} totals",
        f'      <span data-tb-figure="value" data-tb-binding="totals"',
        f'            data-tb-cell="{esc(measure)}" id="val-{esc(measure)}">—</span>.',
        "      Bind prose numbers too; never type one in. Check the file offline",
        "      with <code>tracebi verify --file</code>.</p>",
        "  </header>",
        "",
        '  <div class="tb-grid">',
        f'    <div class="tb-kpi" data-tb-figure="value" data-tb-binding="totals"',
        f'         data-tb-cell="{esc(measure)}" id="kpi-{esc(measure)}">',
        f'      <span class="tb-kpi-label">{esc(m_label)}</span>',
        '      <span class="tb-kpi-value"></span>',
        '      <span class="tb-kpi-context">state the period and comparison</span>',
        "    </div>",
        "  </div>",
    ]
    if layout == "brief":
        parts += _layout_brief_body(measure, dim_ref, table_binding, esc)
    elif layout == "tabbed":
        parts += _layout_tabbed_body(measure, dim_ref, table_binding, esc)
    elif dim_ref:
        d_label = _humanise_label(dim_ref)
        parts += [
            "",
            "  <!-- Chart and the filterable table share one dashboard row.",
            "       tb-cols-2 stacks to a single column on a narrow screen. -->",
            '  <div class="tb-cols-2">',
            '    <div class="tb-card">',
            f"      <h2>{esc(m_label)} by {esc(d_label)}</h2>",
            f'      <div data-tb-figure="chart" data-tb-binding="breakdown"',
            f'           data-tb-type="bar" data-tb-x="{esc(dim_ref)}"',
            f'           data-tb-y="{esc(measure)}" data-tb-value-format="compact"',
            '           id="chart-breakdown"></div>',
            "    </div>",
            '    <div class="tb-card">',
            "      <h2>Detail</h2>",
            '      <p class="tb-note">Filter and search subset which stamped rows',
            "        display — they never compute new numbers. The CSV button",
            "        exports the stamped bytes verbatim.</p>",
            "      <p>",
            "        <label>Search",
            '          <input data-tb-search data-tb-binding="breakdown"',
            '                 placeholder="type to filter…"></label>',
            f"        <label>{esc(d_label)}",
            f'          <select data-tb-filter data-tb-binding="breakdown"',
            f'                  data-tb-column="{esc(dim_ref)}"></select></label>',
            '        <button data-tb-download data-tb-binding="breakdown"',
            '                data-tb-label="Download CSV"></button>',
            "      </p>",
            f'      <table data-tb-figure="table" data-tb-binding="{table_binding}"',
            '             class="tb-table--striped" id="tbl-detail"></table>',
            "    </div>",
            "  </div>",
        ]
    else:
        parts += [
            "",
            '  <div class="tb-card">',
            "    <h2>Detail</h2>",
            f'    <table data-tb-figure="table" data-tb-binding="{table_binding}"',
            '           class="tb-table--striped" id="tbl-detail"></table>',
            "  </div>",
        ]
    parts += [
        "",
        '  <section class="tb-card" data-tb-methodology>',
        "    <h2>How these numbers were made</h2>",
        '    <p class="tb-note">The build appends the transform\'s stated',
        "      methodology (the sink <code>note=</code>) below this line. It is",
        '      the author\'s prose — "the sink satisfied its contract", never',
        '      "the transform was verified".</p>',
        "  </section>",
        "",
        '  <section data-tb-stage="exploration">',
        "    <h3>Working notes</h3>",
        "    <p>Scratch space: this block renders under <code>tracebi dev</code>",
        "       and is DELETED at the final build. Promote what matters into a",
        "       figure above.</p>",
        "  </section>",
        "</main>",
        "</body></html>",
        "",
    ]
    return "\n".join(parts)


def cmd_new_report(args: argparse.Namespace) -> int:
    """
    Scaffold a freeform report package under ``reports/<name>/``.

    Two starter files — ``report.json`` (data bindings) and ``template.html``
    (a page that TEACHES the ``data-tb-*`` figure grammar: bound prose, a KPI, a
    chart, and a filter/search/download table) — bound to the first model in
    ``models/``. No ``script.js``/``style.css``: the runtime draws every figure
    from the stamped bytes, so hand-rolling a parser and a chart (the L0 trap) is
    unnecessary. A directory holding ``report.json`` + ``template.html`` is
    discovered as a report (architecture §7) and built with ``tracebi report
    build <name>``.
    """
    reports_dir: Path = args.reports_dir
    # "finance/Weekly summary" puts the report in a folder: every part before
    # the last is a folder, and the last part is the report's title.
    *folders, title = [p.strip() for p in args.title.split("/")]
    slug = "/".join([_slugify(p) for p in folders if p] + [_slugify(title)])
    args.title = title or args.title
    from tracebi.report_paths import parse_library_mounts
    mounts = parse_library_mounts()
    if mounts is not None:
        # First segment must be a mount label; the rest is under that root.
        parts = slug.split("/", 1)
        if len(parts) < 2:
            print(
                f"refusing to create {slug!r}: with TRACEBI_LIBRARY_MOUNTS "
                f"set, the name must start with a mount label "
                f"({', '.join(lbl for lbl, _ in mounts)})",
                file=sys.stderr,
            )
            return 1
        label, rest = parts
        mount_map = {lbl: path for lbl, path in mounts}
        if label not in mount_map:
            print(
                f"refusing to create {slug!r}: unknown mount label {label!r}. "
                f"Known: {', '.join(sorted(mount_map))}",
                file=sys.stderr,
            )
            return 1
        pkg_dir = mount_map[label] / rest
    else:
        pkg_dir = reports_dir / slug
    if pkg_dir.exists() and not args.force:
        print(f"refusing to overwrite existing {pkg_dir}; pass --force to replace",
              file=sys.stderr)
        return 1
    pkg_dir.mkdir(parents=True, exist_ok=True)

    model, query, note = _scaffold_binding()
    measure = query["measures"][0]
    dims = query.get("dimensions")
    dim_ref = dims[0] if dims else None
    (pkg_dir / "report.json").write_text(
        _report_json_text(args.title, model, query), encoding="utf-8")
    (pkg_dir / "template.html").write_text(
        _report_template_html(args.title, measure, dim_ref, args.layout),
        encoding="utf-8")

    print(f"Created {pkg_dir}/ (report.json, template.html), layout {args.layout}")
    if note:
        print(f"  {note}")
    else:
        print(f"  Bound to model '{model}'. Build it with:")
        print(f"    tracebi report build {slug}")
    return 0


def _resolve_report_target(name: str, reports_dir: Path, *,
                           purpose: str) -> tuple[str, Path]:
    """Resolve *name* to a package directory or a spec file in the library.

    Looks for a package first, then a ``<name>.json`` spec. Returns
    ``("package"|"spec", path)`` or raises ``FileNotFoundError`` listing
    where it looked. When ``TRACEBI_LIBRARY_MOUNTS`` is set, *reports_dir*
    is ignored and the name's first segment selects the mount.
    *purpose* is the read (``view``, ``build``, ``manage``, ``schedule``) the
    seam records for the permission check.
    """
    from tracebi.report_paths import open_report, parse_library_mounts

    # Mounts are authoritative: the name's first segment selects the root.
    # An explicit --reports-dir still wins when mounts are unset.
    kwargs: dict = {"purpose": purpose}
    if parse_library_mounts() is None:
        kwargs["reports_dir"] = reports_dir
    opened = open_report(name, **kwargs)
    if opened.name_error:
        raise FileNotFoundError(opened.name_error)
    if opened.package_dir is not None and opened.has_template:
        return "package", opened.package_dir
    if opened.has_spec:
        return "spec", opened.spec_path
    looked = [p for p in (opened.path, opened.spec_path) if p is not None]
    raise FileNotFoundError(
        f"No report '{name}' found. Looked for a package or spec at:\n  "
        + ("\n  ".join(str(p) for p in looked) if looked else "(no mount matched)")
    )


def _report_pins(name: str, resolve: Optional[str], note: str) -> int:
    """List open pins, or move one into the resolved list.

    Same workbench directory ``collect_state`` reads: ``TRACEBI_WORKBENCH_DIR``
    when set, otherwise ``.tracebi/workbench/<name>``. Writes only ``pins.json``.
    """
    from tracebi.workbench import read_pins, resolve_pin, workbench_dir

    wb = os.environ.get("TRACEBI_WORKBENCH_DIR") or workbench_dir(os.getcwd(), name)
    if resolve:
        try:
            moved = resolve_pin(wb, resolve, note=note)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"resolved {moved.get('id')}")
        return 0
    pins = read_pins(wb)
    if not pins:
        print("no open pins")
        return 0
    for p in pins:
        print(f"{p.get('id')}\t{p.get('kind') or 'pin'}\t{p.get('note') or ''}")
    return 0


def _report_status(kind: str, path: Path, as_json: bool = False) -> int:
    """The earned state of an artifact, from the one workbench state builder.

    What a driving agent and CI call between edits: figures by provenance,
    coverage, unused bindings, pins — printed compact, or as the full state
    JSON with --json.
    """
    if kind != "package":
        print("status applies to artifact packages (reports/<name>/); "
              "verify a spec's receipt with `tracebi verify` instead.",
              file=sys.stderr)
        return 1
    from tracebi.workbench import collect_state

    state = collect_state(str(path), _load_project_models())
    if as_json:
        print(json.dumps(state, indent=2, default=str))
        return 0
    cov = state.get("coverage", {})
    parts = [f"{cov.get('verified', 0)} query-backed",
             f"{cov.get('derived', 0)} python-derived",
             f"{cov.get('unverified', 0)} unverified"]
    if cov.get("unbound_errors"):
        parts.append(f"{cov['unbound_errors']} unbound-ERROR")
    print(f"{state.get('name', path.name)}: {cov.get('total', 0)} figure(s) — "
          + ", ".join(parts))
    for f in state.get("figures", []):
        mark = {"verified": "✓", "derived": "·", "unverified": "·"}.get(
            f.get("provenance"), "✗")
        pin = " 📌" if f.get("pinned") else ""
        print(f"  {mark} {f.get('id')}  [{f.get('provenance')}]"
              f"{'  ← ' + f['binding'] if f.get('binding') else ''}{pin}")
    unused = state.get("unused_bindings") or []
    if unused:
        print(f"  ! unused binding(s): {', '.join(unused)}")
    for p in state.get("pins", []):
        if p.get("kind") == "promote":
            print(f"  → keep: {p.get('request')}")
        elif p.get("kind") == "message":
            print(f"  → message: {p.get('note')}")
    errs = [b for b in state.get("bindings", []) if b.get("error")]
    for b in errs:
        print(f"  ✗ binding '{b['name']}' failed: {b['error']}", file=sys.stderr)
    return 1 if (cov.get("unbound_errors") or errs) else 0


def _snapshot_report_target(kind: str, path: Path, output: Path) -> int:
    """Write a review snapshot — the sendable working state (v2 §2.5).

    Packages only: a spec has no exploration blocks or code to review.
    The snapshot carries NO manifest, on purpose — a weaker-looking receipt
    is worse than none — and ``verify`` refuses the file by name.
    """
    if kind != "package":
        print("snapshot applies to artifact packages (reports/<name>/); "
              "a spec has no exploration state to snapshot — use build.",
              file=sys.stderr)
        return 1
    from tracebi.reports.template_package import TemplatePackage

    models = _load_project_models()
    TemplatePackage(str(path)).snapshot(models, str(output))
    print(f"Snapshot → {output}")
    print("  review copy: exploration blocks kept, code appendix attached, "
          "NO manifest — `tracebi verify` will refuse it by name.")
    return 0


def _build_report_target(kind: str, path: Path, output: Path,
                         theme: Optional[str] = None,
                         badges: bool = False,
                         report_name: Optional[str] = None,
                         filters: Optional[dict] = None) -> Path:
    """Render one report target to *output* (+ a sibling manifest). Returns output.

    *filters*, when set, is applied at build the same way a selection filter
    is: conjoined onto every binding whose model has that dimension.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    models = _load_project_models()
    if kind == "package":
        from tracebi.reports.template_package import TemplatePackage
        TemplatePackage(str(path)).render(
            models, str(output), badges=badges, filters=filters)
    else:
        # A bare .json spec compiles to the artifact package, then renders like
        # a hand-authored one — the same one report form as a package.
        import tempfile

        from tracebi.reports.compile_spec import compile_spec
        from tracebi.reports.template_package import TemplatePackage
        from tracebi.spec import ReportSpec
        spec = ReportSpec.from_json(path.read_text(encoding="utf-8"))
        compiled = compile_spec(
            spec,
            theme_css=_report_theme_css(spec, extra_theme=theme),
            script_js=_report_script_js(spec),
        )
        with tempfile.TemporaryDirectory() as d:
            for fname, content in compiled.files.items():
                (Path(d) / fname).write_text(content, encoding="utf-8")
            TemplatePackage(d).render(
                models, str(output), badges=badges, filters=filters)
    target = report_name or (path.stem if kind == "spec" else path.name)
    from tracebi.state import try_record_report_build
    note = try_record_report_build(
        target, str(output),
        manifest_path=str(output) + ".manifest.json",
    )
    if note:
        print(f"report build was not recorded: {note}", file=sys.stderr)
    return output


def _report_theme_css(spec, extra_theme: Optional[str] = None) -> str:
    """The spec's theme CSS (plus a --theme override file), resolved against
    the reports directory. Later layers win, so the flag stacks last."""
    parts = []
    for name in (getattr(spec, "theme", ""), extra_theme or ""):
        if not name:
            continue
        p = Path(name)
        if not p.is_file():
            p = _default_reports_dir() / name
        if not p.is_file():
            raise FileNotFoundError(
                f"theme file not found: {name} (looked in cwd and "
                f"{_default_reports_dir()})"
            )
        parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _report_script_js(spec) -> str:
    """The spec's script file, resolved against the reports directory."""
    name = getattr(spec, "script", "")
    if not name:
        return ""
    p = Path(name)
    if not p.is_file():
        p = _default_reports_dir() / name
    if not p.is_file():
        raise FileNotFoundError(
            f"script file not found: {name} (looked in cwd and "
            f"{_default_reports_dir()})"
        )
    return p.read_text(encoding="utf-8")


def _serve_file(html_path: Path, name: str, port: int, open_browser: bool) -> None:
    """Serve *html_path*'s directory over HTTP, as ``HTMLRenderer.serve`` does.

    The freeform ``.html`` is already built and self-contained, so preview
    serves that artifact statically rather than re-rendering it — the built-in
    renderers never touch the analyst's ``template.html``.
    """
    import http.server
    import threading
    import webbrowser

    directory = str(html_path.parent.resolve())
    url = f"http://localhost:{port}/{html_path.name}"

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, fmt, *a):  # silence request logs
            pass

    # Bind loopback, not "" (all interfaces): preview serves a directory of
    # rendered reports unauthenticated, and the URL already says localhost.
    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    print(f"\n  TraceBi Report — '{name}'")
    print(f"  Serving at {url}")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    finally:
        server.server_close()


def cmd_report(args: argparse.Namespace) -> int:
    """
    Build, preview, snapshot, or send a report — the build step
    (architecture §2), plus scheduled delivery v1.

        tracebi report build <name>      # → output/<name>.html + manifest
        tracebi report preview <name>    # build, then serve it locally
        tracebi report snapshot <name>   # review snapshot — NO manifest
        tracebi report send <name> --to a@b.com[,c@d]
                                         # build, verify, then email BOTH the
                                         # html and its manifest receipt

    A report is a package (``reports/<name>/``) or a spec
    (``reports/<name>.json``). Output defaults to ``output/<name>.html`` —
    self-contained, offline, and checkable with ``tracebi verify --file``.
    A snapshot keeps the exploration blocks, banners itself, appends a
    read-only code appendix, and writes NO manifest: it is the sendable
    working state, and ``verify`` refuses it by name.

    ``send`` builds exactly like ``build``, then verifies the manifest
    in-process and REFUSES to send while the receipt does not verify —
    distribution never outruns verification. ``--force`` sends anyway with
    the failing verdict pasted prominently into the body: the red flag
    travels WITH the report, never silently.
    """
    reports_dir: Path = args.reports_dir
    if args.action == "status":
        purpose = "view"
    elif args.action == "pins":
        purpose = "manage" if getattr(args, "resolve", None) else "view"
    else:
        purpose = "build"
    try:
        kind, path = _resolve_report_target(args.name, reports_dir, purpose=purpose)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.action == "send" and not getattr(args, "to", None):
        print("report send: --to is required "
              "(one or more addresses, comma-separated)", file=sys.stderr)
        return 1

    if args.action == "pins":
        return _report_pins(args.name, getattr(args, "resolve", None),
                            getattr(args, "note", None) or "")
    if args.action == "status":
        return _report_status(kind, path, as_json=getattr(args, "json", False))
    if args.action == "snapshot":
        output = (Path(args.output) if args.output
                  else Path.cwd() / "output" / f"{args.name}.snapshot.html")
        return _snapshot_report_target(kind, path, output)
    pdf = getattr(args, "format", "html") == "pdf"
    if pdf and args.action != "build":
        print("--format pdf applies to `tracebi report build`.", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else Path.cwd() / "output" / f"{args.name}.html"
    try:
        _build_report_target(kind, path, output,
                             theme=getattr(args, 'theme', None),
                             badges=getattr(args, 'badges', False),
                             report_name=args.name)
    except Exception as exc:  # noqa: BLE001 — a build failure is the user's to fix
        print(f"failed to build report '{args.name}': "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        # A one-line message that names only the exception type leaves the
        # author guessing which binding or column died — the traceback names
        # the figure and the frame, so print it rather than swallowing it. The
        # header above stays for the common bare-eyed read.
        import traceback
        traceback.print_exc()
        return 1

    manifest = output.with_name(output.name + ".manifest.json")
    print(f"Rendered {args.name} ({kind}) → {_project_path(output)}")
    print(f"  manifest → {_project_path(manifest)}")

    if pdf:
        pdf_path = output.with_suffix(".pdf")
        try:
            from tracebi.reports.pdf import print_pdf
            print_pdf(output, pdf_path)
        except (ImportError, RuntimeError) as exc:
            print(f"failed to print '{args.name}' to PDF: {exc}", file=sys.stderr)
            return 1
        print(f"  pdf → {pdf_path}")
        print("  a print of the HTML; it carries no receipt")

    if args.action == "send":
        return _report_send(args, output, manifest)
    if args.action == "preview":
        _serve_file(output, args.name, args.port, not args.no_browser)
    return 0


def _report_send(args: argparse.Namespace, output: Path,
                 manifest_path: Path) -> int:
    """The send action's second half: verification gates distribution.

    The build already happened, exactly as ``build``. The manifest is now
    verified in-process; a receipt that does not verify is refused —
    ``--force`` sends anyway with the failing verdict pasted prominently
    into the body (see :mod:`tracebi._delivery`), so the red flag travels
    WITH the report instead of the report outrunning it.
    """
    from tracebi._delivery import send_report, slack_notify
    from tracebi.verify import load_models, verify_manifest

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"report send: cannot read manifest {manifest_path}: {exc}",
              file=sys.stderr)
        return 1

    models = load_models(args.models_dir)
    try:
        result = verify_manifest(manifest, models)
    except Exception as exc:  # noqa: BLE001 — a corrupt receipt is a user error
        print(f"report send: manifest could not be verified: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ok = result["exit_code"] == 0
    print(result["verdict_detail"], file=sys.stdout if ok else sys.stderr)
    if not ok and not getattr(args, "force", False):
        print(f"refusing to send '{args.name}': the receipt did not verify. "
              f"Distribution never outruns verification; --force sends "
              f"anyway with the verdict pasted into the body.",
              file=sys.stderr)
        return result["exit_code"] or 1

    to = [a.strip() for a in args.to.split(",") if a.strip()]
    try:
        send_report(output, manifest_path, to,
                    subject=getattr(args, "subject", None),
                    verify_result=result)
    except (RuntimeError, OSError) as exc:
        print(f"report send: {exc}", file=sys.stderr)
        return 1
    print(f"Sent {args.name} → {', '.join(to)} (html + manifest receipt)")

    webhook = os.environ.get("TRACEBI_SLACK_WEBHOOK")
    if webhook:
        n = len(result.get("figures") or result.get("sections") or [])
        text = (f"{args.name} delivered · "
                f"{result['verdict'].upper().replace('_', ' ')} · "
                f"{n} figures")
        try:
            slack_notify(webhook, text)
        except Exception as exc:  # noqa: BLE001 — the report already went out
            print(f"slack notify failed (the report was sent): {exc}",
                  file=sys.stderr)
    return 0


# ── Scheduled reports: list / run / serve ───────────────────────────────────

def cmd_schedule(args: argparse.Namespace) -> int:
    """
    Reports that run and deliver themselves, from a package's ``schedule``
    block (see :mod:`tracebi.schedule`).

        tracebi schedule list              # every scheduled package + last run
        tracebi schedule run <name>        # build → verify → deliver → record, now
        tracebi schedule serve             # run every schedule until Ctrl+C

    A ``burst`` block repeats that once per filter value. Each value is its
    own build, with that filter applied, and is emailed to ``burst.to[value]``.
    """
    import getpass

    from tracebi import schedule as sched
    from tracebi.audit import set_actor

    reports_dir: Path = args.reports_dir
    output_dir = Path(args.output_dir)
    from tracebi.report_paths import parse_library_mounts
    if parse_library_mounts() is not None:
        schedules, errors = sched.discover_library_schedules()
    else:
        schedules, errors = sched.discover_schedules(reports_dir)
    for err in errors:
        print(f"skipped {err['report']}: {err['error']}", file=sys.stderr)

    if args.action == "list":
        if getattr(args, "json", False):
            last = sched.last_runs(output_dir)
            print(json.dumps([{**s, "last_run": last.get(s["report"])}
                              for s in schedules], indent=2))
            return 0
        if not schedules:
            print(f"No scheduled reports in {reports_dir}/. Add a \"schedule\" "
                  f"block to a package's report.json, e.g. "
                  f"{{\"cron\": \"0 9 * * MON\", \"to\": [\"a@example.com\"]}}.")
            return 0
        last = sched.last_runs(output_dir)
        for s in schedules:
            print(sched.describe_schedule(s, last.get(s["report"])))
        return 0

    if args.action == "run":
        if not args.name:
            print("schedule run: name the report to run", file=sys.stderr)
            return 1
        match = [s for s in schedules if s["report"] == args.name]
        if not match:
            print(f"schedule run: '{args.name}' has no schedule block in "
                  f"{reports_dir}/{args.name}/report.json", file=sys.stderr)
            return 1
        try:
            set_actor(getpass.getuser(), role="cli")
        except Exception:  # noqa: BLE001 — no controlling user (some containers)
            set_actor(None, role="cli")
        rec = sched.run_schedule(match[0], reports_dir=reports_dir,
                                 output_dir=output_dir,
                                 models_dir=args.models_dir,
                                 send=not args.no_send)
        _print_schedule_run(rec)
        return 0 if rec["status"] in (sched.DELIVERED, sched.BUILT) else 1

    # serve
    if not schedules:
        print(f"No scheduled reports in {reports_dir}/ — nothing to serve.",
              file=sys.stderr)
        return 1

    run = sched.make_job(reports_dir, output_dir, args.models_dir)

    def job(s: dict) -> None:
        _print_schedule_run(run(s))

    try:
        scheduler = sched.build_scheduler(schedules, job, blocking=True)
    except (ImportError, ValueError) as exc:
        print(f"schedule serve: {exc}", file=sys.stderr)
        return 1
    for s in schedules:
        print("  " + sched.describe_schedule(s))
    print(f"\nRunning {len(schedules)} schedule(s). Runs are recorded in "
          f"tracebi_runs (kind=schedule). An existing "
          f"{output_dir / sched.RUN_LOG} is imported once. "
          f"Restart after changing a schedule. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    return 0


def _print_schedule_run(rec: dict) -> None:
    line = f"{rec['report']}: {rec['status']}"
    if rec.get("verdict"):
        line += f" · {rec['verdict']}"
    if rec.get("recipients"):
        line += f" → {', '.join(rec['recipients'])}"
    ok = rec["status"] in ("delivered", "built")
    out = sys.stdout if ok else sys.stderr
    print(line, file=out)
    if rec.get("error"):
        print(f"  {rec['error']}", file=out)
    for sl in rec.get("slices") or []:
        piece = f"  {sl.get('value')}: {sl.get('status')}"
        if sl.get("recipients"):
            piece += f" → {', '.join(sl['recipients'])}"
        if sl.get("output"):
            piece += f"  {sl['output']}"
        print(piece, file=out)
        if sl.get("error"):
            print(f"    {sl['error']}", file=out)


# ── Workbench sessions: export / clear ──────────────────────────────────────

def cmd_agent(args: argparse.Namespace) -> int:
    """Summarize the gateway call log (``TRACEBI_MCP_LOG=1``)."""
    from tracebi import _gateway_log

    try:
        since = _gateway_log.parse_since(args.since) if args.since else None
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not os.path.isfile(_gateway_log.LOG_PATH):
        print(f"No gateway log at {_gateway_log.LOG_PATH}. Set TRACEBI_MCP_LOG=1 "
              "before starting `tracebi mcp` to record one.", file=sys.stderr)
        return 1
    summary = _gateway_log.summarize(
        _gateway_log.read_lines(_gateway_log.LOG_PATH, since))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(_gateway_log.format_summary(summary))
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    """
    Save or reset a workbench session feed.

        tracebi session export [name]    # → explorations/<name>.html — ONE living record
        tracebi session clear [name]     # remove the feed and pins

    *name* defaults to the discovery session (``_discovery``); a package
    name addresses that package's session. The export is an exploration
    RECORD — the full feed, uncapped, as one self-contained lab-notebook
    HTML with NO manifest; ``tracebi verify`` refuses it by name. ``clear``
    is prompt-free, but refuses while a dev server's heartbeat for the
    session is fresh — the live watcher would keep posting over the reset.
    """
    import time

    from tracebi.workbench import (
        ACTIVE_FILE, DISCOVERY_NAME, EXHIBITS_FILE, HEARTBEAT_WINDOW,
        PINS_FILE,
    )

    name = args.name or DISCOVERY_NAME
    wb = os.path.join(os.getcwd(), ".tracebi", "workbench", name)

    if args.action == "export":

        from tracebi._session_export import export_session

        ext = "md" if getattr(args, "format", "html") == "md" else "html"
        # ONE living record per session, not a dated diary: the exploration
        # is the base document that evolves into the pipeline, so re-exports
        # overwrite the same file and git carries the timeline — each
        # iteration is a readable diff (the .md twin especially).
        fname = name.lstrip("_") or name
        out = Path(args.output) if args.output else (
            Path("explorations") / f"{fname}.{ext}")
        existed = out.exists()
        try:
            export_session(wb, str(out), title=args.title)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"{'Updated' if existed else 'Exported'} session '{name}' "
              f"→ {out}")
        print("  The living exploration record — re-export as it evolves; "
              "git is its timeline. verify refuses it by name.")
        return 0

    # clear
    try:
        age = time.time() - os.path.getmtime(os.path.join(wb, ACTIVE_FILE))
    except OSError:
        age = None
    if age is not None and age <= HEARTBEAT_WINDOW:
        wait = max(1, int(HEARTBEAT_WINDOW - age) + 1)
        print(f"session '{name}' has a fresh dev-server heartbeat "
              f"({int(age)}s old). If the server is still running, stop it; "
              f"the heartbeat then goes stale after {HEARTBEAT_WINDOW}s, so if "
              f"you just stopped it, retry in ~{wait}s.", file=sys.stderr)
        return 1
    removed = [fname for fname in (EXHIBITS_FILE, PINS_FILE)
               if os.path.isfile(os.path.join(wb, fname))]
    for fname in removed:
        os.remove(os.path.join(wb, fname))
    if removed:
        print(f"Cleared session '{name}' ({', '.join(removed)}).")
    else:
        print(f"Session '{name}' has no feed or pins to clear.")
    return 0


# ── Argparse wiring ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracebi",
        description=(
            "TraceBi — ask it, build it, schedule it: analysts and agents "
            "write every kind of report from one set of approved definitions. "
            "Every number carries a receipt you can re-check."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tracebi {_tracebi_version()}",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_default_models_dir(),
        help="Directory holding model definitions (default: ./models).",
    )
    parser.add_argument(
        "--pipelines-dir",
        type=Path,
        default=_default_pipelines_dir(),
        help="Directory holding pipeline definitions (default: ./pipelines).",
    )
    parser.add_argument(
        "--connections-dir",
        type=Path,
        default=_default_connections_dir(),
        help="Directory holding connection files (default: ./connections).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser(
        "init",
        help="Scaffold a complete three-phase project (inputs/, transforms/, "
             "models/, reports/, …) whose sample loop ends in "
             "`tracebi verify` reading REPRODUCES.",
    )
    p_init.add_argument("project", help="Target directory name.")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite existing files.")
    p_init.add_argument(
        "--template",
        default=None,
        metavar="NAME",
        help="Scaffold a named starter instead of the sample dashboard. "
             "Known: saas-metrics (also saas_metrics), sales-pipeline "
             "(also sales_pipeline). Unknown names are refused with the "
             "known list.",
    )
    p_init.set_defaults(func=cmd_init)

    p_spec = sub.add_parser(
        "spec",
        help="Work with report specs (a report as JSON): print the schema, "
             "validate a spec without running it, or render one.",
    )
    p_spec.add_argument("action", choices=["schema", "validate", "render"])
    p_spec.add_argument(
        "--theme",
        help="Extra CSS file stacked over the spec's own theme (later wins).",
    )
    p_spec.add_argument("file", nargs="?", help="Path to a spec .json file.")
    p_spec.add_argument("--output", help="Output path for `render`.")
    p_spec.set_defaults(func=cmd_spec)

    p_context = sub.add_parser(
        "context",
        help="Print the framework's vocabulary as JSON (section and panel "
             "types, DataSet verbs, measures, operators, conventions).",
    )
    p_context.add_argument("--model", help="Also include this model's schema.")
    p_context.add_argument(
        "--brief", action="store_true",
        help="The token-lean tier (~40%% of the payload): semantic model, "
             "figure grammar, contracts, conventions — everything the "
             "package-first loop needs. Omits the legacy section classes, "
             "DataSet verbs, and Python cheat sheets.",
    )
    p_context.add_argument("--compact", action="store_true",
                           help="Single-line JSON.")
    p_context.set_defaults(func=cmd_context)

    p_knowledge = sub.add_parser(
        "knowledge",
        help="Analyst good-practice lessons. No arg lists the curriculum; "
             "'knowledge <slug>' prints one lesson.",
    )
    p_knowledge.add_argument("slug", nargs="?",
                             help="Lesson slug to print in full.")
    p_knowledge.set_defaults(func=cmd_knowledge)

    p_serve = sub.add_parser(
        "serve",
        help="Serve this project's web UI (models, reports, pipelines) "
             "at http://127.0.0.1:8000.",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true",
                         help="Restart on file changes (development).")
    p_serve.set_defaults(func=cmd_serve)

    p_login = sub.add_parser(
        "login",
        help="Sign in to a TraceBi server (opens your browser; the company "
             "login). Tokens are kept in ~/.config/tracebi/credentials.json.",
    )
    p_login.add_argument("--server", required=True, metavar="URL",
                         help="The server's address, such as https://bi.example.com.")
    p_login.add_argument("--timeout", type=float, default=300,
                         help="Seconds to wait for the browser (default 300).")
    p_login.set_defaults(func=cmd_login)

    p_logout = sub.add_parser(
        "logout", help="Revoke this machine's sign-in to a TraceBi server and forget it.")
    p_logout.add_argument("--server", required=True, metavar="URL")
    p_logout.set_defaults(func=cmd_logout)

    p_update = sub.add_parser(
        "update",
        help="Check for a newer TraceBi and show (or run) the command that "
             "updates this install.",
    )
    p_update.add_argument("--check", action="store_true",
                          help="Only report; never run the update.")
    p_update.add_argument("--yes", "-y", action="store_true",
                          help="Run the update without asking (pip installs).")
    p_update.set_defaults(func=cmd_update)

    p_dev = sub.add_parser(
        "dev",
        help="Build in the web app, in dev mode (loopback only). An artifact "
             "package (reports/<name>/) opens with Build mode on: the report "
             "re-renders as your agent saves, a timeline shows what changed, "
             "you point at a figure, and Data and Checks show what it reads. "
             "With no name it opens the app where what your agent shows you "
             "lands, before any report exists. --classic keeps the old "
             "stdlib preview server.",
    )
    p_dev.add_argument("name", nargs="?",
                       help="Package name under reports/. Omit to open the app "
                            "before any report is open.")
    p_dev.add_argument("--port", type=int, default=8001,
                       help="Port to serve on (default 8001).")
    p_dev.add_argument("--no-browser", action="store_true",
                       help="Do not open the browser automatically.")
    mode = p_dev.add_mutually_exclusive_group()
    mode.add_argument("--classic", action="store_true",
                      help="Serve the classic preview server and its workbench "
                           "page (/__workbench) instead of the app: the fallback "
                           "when the web extra is not installed, and what "
                           "tracebi dev was before the app.")
    mode.add_argument("--app", action="store_true",
                      help="Open the app. This is the default; the flag is kept "
                           "so older commands still work.")
    p_dev.set_defaults(func=cmd_dev)

    p_validate = sub.add_parser(
        "validate",
        help="Sanity-check the project: layout, plus load every model in "
             "models/ and verify its dimension keys are unique.",
    )
    p_validate.set_defaults(func=cmd_validate)

    p_connect = sub.add_parser(
        "connect",
        help="Connect a warehouse you already have (postgres, snowflake, "
             "bigquery, duckdb): test it, write the secret to .env, and "
             "write connections/<name>.yaml.",
    )
    p_connect.add_argument("name", help="Connection name, letters, digits and underscore (e.g. wh).")
    p_connect.add_argument(
        "--kind", choices=["postgres", "snowflake", "bigquery", "duckdb"],
        help="Warehouse kind. Prompted when stdin is a terminal.",
    )
    p_connect.add_argument("--url", help="Postgres SQLAlchemy URL (postgresql://...).")
    p_connect.add_argument("--account", help="Snowflake account identifier.")
    p_connect.add_argument("--user", help="Snowflake user.")
    p_connect.add_argument("--password", help="Snowflake password. Never printed.")
    p_connect.add_argument("--warehouse", help="Snowflake warehouse.")
    p_connect.add_argument("--database", help="Snowflake database, or DuckDB file path.")
    p_connect.add_argument("--schema", help="Snowflake schema.")
    p_connect.add_argument("--role", help="Snowflake role. Optional.")
    p_connect.add_argument("--project", help="BigQuery GCP project id.")
    p_connect.add_argument("--dataset", help="BigQuery dataset.")
    p_connect.add_argument(
        "--credentials",
        help="Path to a BigQuery service-account JSON file. Optional.",
    )
    p_connect.add_argument(
        "--test", action=argparse.BooleanOptionalAction, default=True,
        help="Connect and list tables before writing anything (default). "
             "--no-test writes without connecting.",
    )
    p_connect.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing .env key or connection file.",
    )
    p_connect.add_argument(
        "--python", action="store_true",
        help="Write the legacy models/_connections/<name>.py module instead "
             "of connections/<name>.yaml.",
    )
    p_connect.set_defaults(func=cmd_connect)

    p_new_model = sub.add_parser(
        "new-model",
        help="Scaffold a new model definition, or draft one from a connection.",
    )
    p_new_model.add_argument("title", help='Free-form title, e.g. "Sales Model".')
    p_new_model.add_argument("--force", action="store_true", help="Overwrite if exists.")
    p_new_model.add_argument(
        "--from", dest="from_connection", metavar="CONNECTION",
        help="Draft the model from this `tracebi connect` connection. "
             "Metadata only; no row scan.",
    )
    p_new_model.add_argument(
        "--tables",
        help="Comma-separated tables to draft. Required with --from.",
    )
    p_new_model.add_argument(
        "--python", action="store_true",
        help="Write a Python models/<name>.py instead of the YAML default.",
    )
    p_new_model.set_defaults(func=cmd_new_model)

    p_import = sub.add_parser(
        "import",
        help="Draft a TraceBi model from an existing tool.",
    )
    imported = p_import.add_subparsers(dest="import_cmd", required=True)
    p_import_dbt = imported.add_parser(
        "dbt",
        help="Draft a model from a dbt manifest.json (does not run dbt).",
        description="Draft models/<name>.py from a dbt manifest.json. "
                    "This command does not run dbt and does not load .env.",
    )
    p_import_dbt.add_argument(
        "path",
        type=Path,
        help="dbt project root (contains target/manifest.json) or a manifest.json path.",
    )
    p_import_dbt.add_argument(
        "--name",
        help="Model name. Default: the dbt project name, else the folder name.",
    )
    p_import_dbt.add_argument(
        "--connection",
        help="Wire models/_connections/<name>.py from `tracebi connect`. "
             "Refused if that module is missing.",
    )
    p_import_dbt.add_argument(
        "--schema",
        help="Only include dbt models in this schema. "
             "Default: every non-ephemeral model.",
    )
    p_import_dbt.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing model file.",
    )
    p_import_dbt.set_defaults(func=cmd_import_dbt)

    p_list_models = sub.add_parser("list-models", help="List model definition files.")
    p_list_models.set_defaults(func=cmd_list_models)

    p_warehouse = sub.add_parser(
        "warehouse",
        help="List warehouse tables, or one table's columns, from metadata.",
    )
    wh = p_warehouse.add_subparsers(dest="warehouse_cmd", required=True)
    p_wh_tables = wh.add_parser(
        "tables",
        help="List tables, or describe one table's columns and types.",
    )
    p_wh_tables.add_argument(
        "--connector", default="",
        help="Limit to this connector name.",
    )
    p_wh_tables.add_argument(
        "--table", default="",
        help="Describe this table's columns instead of listing tables.",
    )
    p_wh_tables.add_argument(
        "--json", action="store_true",
        help="Print the structured result as JSON.",
    )
    p_wh_tables.set_defaults(func=cmd_warehouse)

    p_new_pipeline = sub.add_parser("new-pipeline", help="Scaffold a new pipeline definition.")
    p_new_pipeline.add_argument("title", help='Free-form title, e.g. "Sales Pipeline".')
    p_new_pipeline.add_argument("--force", action="store_true", help="Overwrite if exists.")
    p_new_pipeline.add_argument(
        "--python", action="store_true",
        help="Write a Python pipelines/<name>.py instead of the YAML default.",
    )
    p_new_pipeline.set_defaults(func=cmd_new_pipeline)

    p_run_transform = sub.add_parser(
        "run-transform",
        help="Execute a phase-① transform (.py or .ipynb) top-to-bottom in "
             "a fresh namespace — the sink never comes from out-of-order "
             "kernel state. Notebook-shaped .py (# %% cells) and literal "
             ".ipynb both work.",
    )
    p_run_transform.add_argument(
        "name", help="Transform name under transforms/ (suffix optional).")
    p_run_transform.set_defaults(func=cmd_run_transform)

    p_new_transform = sub.add_parser(
        "new-transform",
        help="Scaffold a phase-① transform (transforms/<name>.py): pandas "
             "that reads a raw input and sinks star tables to the warehouse.",
    )
    p_new_transform.add_argument("title", help='Free-form title, e.g. "Orders Clean".')
    p_new_transform.add_argument("--force", action="store_true", help="Overwrite if exists.")
    p_new_transform.add_argument(
        "--transforms-dir", type=Path, default=_default_transforms_dir(),
        help="Directory holding transforms (default: ./transforms).",
    )
    p_new_transform.set_defaults(func=cmd_new_transform)

    p_list_pipelines = sub.add_parser("list-pipelines", help="List pipeline definition files.")
    p_list_pipelines.set_defaults(func=cmd_list_pipelines)

    p_new_report = sub.add_parser(
        "new-report",
        help="Scaffold a freeform report package (reports/<name>/): report.json "
             "bindings + a template.html that teaches the data-tb-* figure grammar.",
    )
    p_new_report.add_argument(
        "title", help='Free-form title, e.g. "Portfolio Book". Put it in a folder '
                      'with a path: "Finance/Month end/Close pack" creates '
                      'reports/finance/month_end/close_pack/.')
    p_new_report.add_argument("--force", action="store_true", help="Overwrite if exists.")
    p_new_report.add_argument("--reports-dir", type=Path, default=_default_reports_dir(),
                              help="Directory holding report packages (default: ./reports).")
    p_new_report.add_argument(
        "--layout",
        choices=["brief", "dashboard", "tabbed"],
        default="dashboard",
        help="Page structure. brief: answer sentence, KPIs, one chart. "
             "dashboard (default): brief, then a chart beside a filterable "
             "table. tabbed: the same header, then Overview and Detail tabs.",
    )
    p_new_report.set_defaults(func=cmd_new_report)

    p_report = sub.add_parser(
        "report",
        help="Build a report package or spec to one self-contained .html + "
             "manifest, preview it locally, or send it (build → verify → "
             "email with the receipt attached).",
    )
    p_report.add_argument("action",
                          choices=["build", "preview", "snapshot", "status",
                                   "pins", "send"])
    p_report.add_argument("name", help="Report name (package dir or spec stem).")
    p_report.add_argument(
        "--to",
        help="With `send`: recipient address(es), comma-separated. Send "
             "builds like `build`, verifies the manifest in-process, and "
             "refuses to send unless the receipt verifies (--force "
             "overrides, pasting the verdict into the body). Email needs "
             "TRACEBI_SMTP_URL (smtp://user:pass@host:port or smtps://) "
             "and TRACEBI_SMTP_FROM; TRACEBI_SLACK_WEBHOOK adds a Slack "
             "ping. To send on a schedule, add a \"schedule\" block to "
             "report.json and use `tracebi schedule` (serve, or call "
             "`tracebi schedule run <name>` from cron).",
    )
    p_report.add_argument(
        "--subject",
        help="With `send`: email subject (default: '[tracebi] <report name>').",
    )
    p_report.add_argument(
        "--force", action="store_true",
        help="With `send`: send even when the receipt does not verify. The "
             "failing verdict is pasted prominently into the email body — "
             "a red flag travels WITH the report, never silently.",
    )
    p_report.add_argument("--output", help="Output .html path (default: output/<name>.html).")
    p_report.add_argument(
        "--format", choices=["html", "pdf"], default="html",
        help="With `build`: html (default), or pdf to also write a sibling "
             ".pdf — a print of the built HTML. The PDF carries no receipt; "
             "the HTML and manifest stay the checkable artifact. Needs "
             "pip install 'tracebi[pdf]' and "
             "python -m playwright install chromium.",
    )
    p_report.add_argument(
        "--resolve",
        metavar="ID",
        help="With `pins`: move that open pin into the resolved list, "
             "keeping it with a timestamp, the current actor, and --note.",
    )
    p_report.add_argument(
        "--note",
        help="With `pins --resolve`: the one-line note of what was done.",
    )
    p_report.add_argument(
        "--json", action="store_true",
        help="With `status`: print the full workbench state as JSON (what "
             "the workbench page and the MCP workbench_state tool consume).",
    )
    p_report.add_argument(
        "--badges", action="store_true",
        help="Render the per-figure provenance badges on the page. Off by "
             "default — the receipt drawer carries provenance in one place. "
             "The manifest is unaffected either way.",
    )
    p_report.add_argument(
        "--theme",
        help="Extra CSS file stacked over the spec's own theme (later wins). "
             "Spec targets only; packages carry their own style.css.",
    )
    p_report.add_argument("--reports-dir", type=Path, default=_default_reports_dir(),
                          help="Directory holding report packages (default: ./reports).")
    p_report.add_argument("--port", type=int, default=8080,
                          help="Port for `preview` (default 8080).")
    p_report.add_argument("--no-browser", action="store_true",
                          help="With `preview`, do not open the browser.")
    p_report.set_defaults(func=cmd_report)

    p_schedule = sub.add_parser(
        "schedule",
        help="Reports that run and deliver themselves: list the packages with "
             "a report.json \"schedule\" block, run one now (build → verify "
             "→ email → record), or serve every schedule until Ctrl+C. "
             "A burst block builds once per filter value, each with its own "
             "recipients (burst.to[value]); a value missing from burst.to is "
             "skipped and recorded, and is not sent to the top-level to.",
    )
    p_schedule.add_argument("action", choices=["list", "run", "serve"])
    p_schedule.add_argument("name", nargs="?",
                            help="With `run`: the report package to run.")
    p_schedule.add_argument(
        "--no-send", action="store_true",
        help="With `run`: build and verify, record the run, send nothing.")
    p_schedule.add_argument("--json", action="store_true",
                            help="With `list`: print JSON, with each last run.")
    p_schedule.add_argument("--reports-dir", type=Path, default=_default_reports_dir(),
                            help="Directory holding report packages (default: ./reports).")
    p_schedule.add_argument(
        "--output-dir", default="output",
        help="Where runs write the artifact and its manifest. An existing "
             "schedule_runs.jsonl here is imported into the run store once "
             "(default: ./output).")
    p_schedule.set_defaults(func=cmd_schedule)

    p_session = sub.add_parser(
        "session",
        help="Save or reset a workbench session feed. `session export "
             "[name]` renders the FULL feed (uncapped) to one exploration-"
             "record HTML — a lab notebook, not a report: no manifest, and "
             "`tracebi verify` refuses it by name. `session clear [name]` "
             "removes the feed and pins (refused while a dev server's "
             "heartbeat is fresh). Name defaults to the discovery session.",
    )
    p_session.add_argument("action", choices=["export", "clear"])
    p_session.add_argument(
        "name", nargs="?", default=None,
        help="Session name: a package name, or omit for the discovery "
             "session (_discovery).",
    )
    p_session.add_argument(
        "-o", "--output",
        help="Output .html path for `export` "
             "(default: explorations/<name>.<ext> — one living record per "
             "session; re-exports overwrite it and git carries the "
             "timeline).",
    )
    p_session.add_argument(
        "--title",
        help="Page title (default: '<session> — exploration record').",
    )
    p_session.add_argument(
        "--format", choices=["html", "md"], default="html",
        help="Export format: self-contained HTML (default; charts render, "
             "verify refuses it by name) or the Markdown twin — the "
             "git-review format: raw markdown notes verbatim, tables for "
             "frames and sketches, made for reading in a pull request. An "
             "explicit -o extension (.md/.html) also selects the format.",
    )
    p_session.set_defaults(func=cmd_session)

    p_agent = sub.add_parser(
        "agent",
        help="`agent log` summarizes the opt-in gateway call log "
             "(.tracebi/gateway_log.jsonl, written when TRACEBI_MCP_LOG=1): "
             "calls and error rate per tool, the top ten errors, and the "
             "first-build success rate.",
    )
    p_agent.add_argument("action", choices=["log"])
    p_agent.add_argument(
        "--since", metavar="AGE",
        help="Only calls newer than AGE, like 7d, 12h or 30m.")
    p_agent.add_argument(
        "--json", action="store_true", help="Print the summary as JSON.")
    p_agent.set_defaults(func=cmd_agent)

    p_migrate = sub.add_parser(
        "migrate",
        help="Migrate legacy report forms to the artifact package. "
             "`migrate spec <file.json>` compiles a JSON spec into "
             "reports/<stem>/ (report.json bindings + template.html of "
             "default-component figures) alongside the original.",
    )
    p_migrate.add_argument("what", choices=["spec", "model", "connection", "pipeline"])
    p_migrate.add_argument(
        "path", help="Path to a reports/<name>.json spec, a models/<name>.py "
                     "model, a models/_connections/<name>.py connection, or a "
                     "pipelines/<name>.py pipeline.")
    p_migrate.add_argument("--force", action="store_true",
                           help="Overwrite an existing target directory (spec) "
                                "or models/<name>.yaml (model).")
    p_migrate.add_argument("--write", action="store_true",
                           help="model, connection, pipeline: write the .yaml "
                                "beside the project's other files instead of "
                                "printing it. The Python file is never deleted.")
    p_migrate.set_defaults(func=cmd_migrate)

    p_run_pipeline = sub.add_parser(
        "run-pipeline",
        help="Run a pipeline's layers (the execution plane — no web server needed).",
    )
    p_run_pipeline.add_argument("name", help="Pipeline name, as listed by list-pipelines.")
    p_run_pipeline.add_argument(
        "--layer",
        help="Run only this layer instead of the whole pipeline.",
    )
    p_run_pipeline.add_argument(
        "--refresh", action="store_true",
        help="With --layer, run its upstream chain first.",
    )
    p_run_pipeline.add_argument(
        "--status", action="store_true",
        help="Show each layer and its last run without executing anything.",
    )
    p_run_pipeline.set_defaults(func=cmd_run_pipeline)

    p_mcp = sub.add_parser(
        "mcp",
        help="Serve the agent gateway over the Model Context Protocol. "
             "`mcp config` prints a client snippet instead of serving.",
    )
    p_mcp.add_argument(
        "mcp_action", nargs="?", choices=("config",),
        help="config — print the MCP client snippet and exit.",
    )
    p_mcp.add_argument(
        "--client", choices=_MCP_CLIENTS,
        help="With `mcp config`: claude-code, cursor, or claude-desktop.",
    )
    p_mcp.add_argument(
        "--http", metavar="URL",
        help="With `mcp config`: streamable-HTTP snippet for URL "
             "(claude-code or cursor). The Authorization header is a "
             "placeholder — ${TRACEBI_MCP_TOKEN} for Claude Code, "
             "${env:TRACEBI_MCP_TOKEN} for Cursor — never a token value. "
             "Refused for claude-desktop.",
    )
    p_mcp.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio",
        help="stdio for a local agent (default); http for a remote one.",
    )
    p_mcp.add_argument(
        "--host", default="127.0.0.1",
        help="Address for --transport http (default 127.0.0.1). "
             "Use 0.0.0.0 inside a container. A non-loopback host with "
             "--insecure also needs --allow-insecure-bind.",
    )
    p_mcp.add_argument(
        "--port", type=int, default=8765,
        help="Port for --transport http (default 8765).",
    )
    p_mcp.add_argument(
        "--insecure", action="store_true",
        help="Serve --transport http without authentication. Deliberate "
             "opt-out: without it, http requires TRACEBI_MCP_TOKEN.",
    )
    p_mcp.add_argument(
        "--allow-insecure-bind", action="store_true",
        help="Allow --insecure on a non-loopback --host. Anyone who can "
             "reach the port gets full query access.",
    )
    p_mcp.set_defaults(func=cmd_mcp)

    p_verify = sub.add_parser(
        "verify",
        help="Re-run every recorded query in a rendered manifest and "
             "classify each section: reproduces, source drift, model changed, "
             "unexplained, or unverifiable. Exits 0 only when something was actually "
             "checked and nothing failed.",
    )
    p_verify.add_argument(
        "manifest", nargs="?", default=None,
        help="Path to a *.manifest.json file (query → model check). "
             "Omit when using --file.",
    )
    # A wholly separate offline check: the shipped .html's embedded bytes vs
    # the manifest (architecture §3.2). Kept apart from the positional so
    # neither can be mistaken for the other.
    p_verify.add_argument(
        "--strict", action="store_true",
        help="Fail unless every FIGURE in a schema-2 manifest reproduces — "
             "the CI gate for a finalized report.",
    )
    p_verify.add_argument(
        "--file", dest="verify_file", default=None,
        help="Path to a self-contained report *.html*; rehash its embedded "
             "data against a sibling <file>.manifest.json (no model needed).",
    )
    p_verify.add_argument(
        "--contracts", action="store_true",
        help="Also re-run the warehouse's recorded sink contracts against "
             "the current warehouse AND compare the certified per-table "
             "fingerprints to the data now (catching an out-of-band change "
             "that still passes the checks). A separate claim, reported "
             "separately: it says whether today's sink still satisfies its "
             "declared contract — never that the transform was verified.",
    )
    p_verify.add_argument(
        "--manifest", dest="file_manifest", default=None,
        help="With --file: the manifest to check against, if it is not the "
             "sibling <file>.manifest.json.",
    )
    # Distinct dest: a subparser option sharing dest with a main-parser
    # option would clobber an already-parsed `tracebi --models-dir X verify`.
    p_verify.add_argument(
        "--models-dir", type=Path, default=None, dest="verify_models_dir",
        help="Directory holding model definitions (default: ./models).",
    )
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
