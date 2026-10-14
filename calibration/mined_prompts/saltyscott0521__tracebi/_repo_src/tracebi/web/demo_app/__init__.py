import os as _os

from tracebi import model_registry as _model_registry

# The demo models ship inside this package (models/). Register them before any
# module in the package resolves them by name — pipeline.py and the report
# files call get_model("wealth_model") at import time, and every import path
# into the package runs this __init__ first.
_model_registry.auto_discover(_os.path.join(_os.path.dirname(__file__), "models"))

from tracebi.web.demo_app import registry as _  # noqa: F401,E402
