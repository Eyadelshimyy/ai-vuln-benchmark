"""
Demo app registry — the single wiring file for the demo app instance.

Everything registered here is visible to the web UI. The demo app is fully
self-contained: its DataModels live in this package's models/ subdirectory
(wealth_model.py, MemoryConnector-backed), the sales model that reads its
pipeline's cleaned tables (pipeline.py), its reports in reports/, and its
pipelines below. It runs identically from any working
directory, which is also what makes it a faithful miniature of a real
project: models/ + reports/ + a pipeline, wired through the one registry.

To add a new report: create reports/<name>.py with a
@register.report(...) decorated factory function. It will be picked up
automatically on the next server start (or dev-mode reload).
"""

import os

from tracebi.web.api.registry import registry
from tracebi import model_registry
from tracebi.model_registry import get_model
from tracebi.web.demo_app.pipeline import runner, sales_model
from tracebi.web.demo_app.wealth_pipeline import runner as wealth_runner
from tracebi.web.discovery import auto_discover

# ── Models, all named <thing>_model ───────────────────────────────────────────
# wealth_model lives in this package's models/, which the package __init__
# registers with the shared model registry under its file stem, so
# get_model("wealth_model") resolves regardless of cwd.
#
# sales_model reads the silver tables the sales pipeline writes, so it is
# defined beside that pipeline (pipeline.py). Register it by name here: the
# sales reports resolve it as "sales_model" at render time.

wealth_model = get_model("wealth_model")

model_registry.register(sales_model)
registry.add_model(sales_model, default=True)
registry.add_model(wealth_model)

# Surface each model's connectors on the Sources page.
for _conn in (*sales_model.connectors(), *wealth_model.connectors()):
    registry.add_connector(_conn)

# ── Pipeline ──────────────────────────────────────────────────────────────────

registry.add_pipeline("sales", runner)
registry.add_pipeline("wealth", wealth_runner)

# ── Reports (auto-discovered) ─────────────────────────────────────────────────
# Each .py file in reports/ that is not prefixed with _ is imported.
# The @register.report(...) decorator in each file fires on import,
# registering the factory with the registry above.

_reports_dir = os.path.join(os.path.dirname(__file__), "reports")
auto_discover(_reports_dir)
